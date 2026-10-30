"""Word-retrieval adaptations; upstream sources and licenses live in third_party/."""
import ast
from dataclasses import asdict, dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import typing as tp

import torch
from torch import nn
from torch.nn import functional as F


SOURCE = Path(os.environ.get('NEUROMOMENT_BASELINE_SOURCES', str(Path(__file__).resolve().parents[1] / 'third_party/baseline_sources')))


@dataclass
class BaselineConfig:
    arch: str
    channels: int
    text_dim: int
    samples: int = 154
    sample_rate: float = 128.
    subjects: tuple = ()
    positions: tuple = ()
    brain_width: int = 320
    brain_depth: int = 10


class EEGNet(nn.Module):
    """EEGNet-8,2: native pooling, time kernels adapted to the configured sampling rate, word projection."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        kernel = round(config.sample_rate / 2)
        separable = round(16 * config.sample_rate / 128)
        self.temporal_pad = (kernel // 2 - (kernel % 2 == 0), kernel // 2, 0, 0)
        self.separable_pad = (separable // 2 - (separable % 2 == 0), separable // 2, 0, 0)
        self.temporal = nn.Conv2d(1, 8, (1, kernel), bias=False)
        self.temporal_bn = nn.BatchNorm2d(8, eps=1e-3, momentum=.01)
        self.spatial = nn.Conv2d(8, 16, (config.channels, 1), groups=8, bias=False)
        self.spatial_bn = nn.BatchNorm2d(16, eps=1e-3, momentum=.01)
        self.depthwise = nn.Conv2d(16, 16, (1, separable), groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, 1, bias=False)
        self.separable_bn = nn.BatchNorm2d(16, eps=1e-3, momentum=.01)
        self.dropout = nn.Dropout(.5)
        self.project = nn.Linear(16 * (config.samples // 4 // 8), config.text_dim)
        if config.samples < 32:
            raise ValueError('EEGNet input too short for native pooling')

    @torch.no_grad()
    def constrain(self):
        weight = self.spatial.weight
        norm = weight.flatten(1).norm(dim=1).clamp_min(1e-12)
        weight.mul_((norm.clamp_max(1.) / norm)[:, None, None, None])

    def forward(self, x, subjects=None):
        x = self.temporal_bn(self.temporal(F.pad(x[:, None], self.temporal_pad)))
        x = F.elu(self.spatial_bn(self.spatial(x)))
        x = self.dropout(F.avg_pool2d(x, (1, 4)))
        x = self.pointwise(self.depthwise(F.pad(x, self.separable_pad)))
        x = self.dropout(F.avg_pool2d(F.elu(self.separable_bn(x)), (1, 8)))
        return F.normalize(self.project(x.flatten(1)), dim=-1)


class FixedPositionGetter:
    """Same normalized MNE-layout coordinates for every record in each dataset."""
    def get_positions(self, batch):
        return batch.positions

    def is_invalid(self, positions):
        return (positions == -.1).all(-1)


def upstream_classes():
    """Execute selected unmodified upstream class bodies without unrelated dependencies.

    Inactive STFT/RNN/dropout options of SimpleConv are deliberately not exposed.
    SubjectLayers, ChannelMerger and ConvSequence execute the author's exact code.
    """
    namespace = dict(torch=torch, nn=nn, F=F, tp=tp, partial=partial, math=math,
                     PositionGetter=FixedPositionGetter, __name__=__name__)
    for filename, names in [
        ('brainmagick_bm_models_common.py', {'SubjectLayers', 'LayerScale', 'ConvSequence', 'FourierEmb', 'ChannelMerger'}),
        ('brainmagick_bm_models_simpleconv.py', {'SimpleConv'}),
    ]:
        path = SOURCE / filename
        if not path.exists():
            raise FileNotFoundError('Optional BrainModule sources are missing. Run python scripts/fetch_brainmodule.py; see THIRD_PARTY_NOTICES.md for its CC-BY-NC-4.0 license.')
        provenance = json.loads(path.with_suffix(path.suffix + '.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != provenance['sha256']:
            raise ValueError(f'Upstream source hash mismatch: {path}')
        tree = ast.parse(path.read_text())
        selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
        if len(selected) != len(names):
            raise ValueError('Missing upstream class')
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


class BrainModule(nn.Module):
    """Author's clip_conv SimpleConv, mean-pool its word-embedding output in time."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        positions = torch.tensor(config.positions, dtype=torch.float32)
        if positions.shape != (config.channels, 2) or not config.subjects:
            raise ValueError('BrainModule needs channel positions and training subject identities')
        self.register_buffer('positions', positions)
        self.encoder = upstream_classes()['SimpleConv'](
            in_channels={'meg': config.channels}, out_channels=config.text_dim,
            hidden={'meg': config.brain_width}, depth=config.brain_depth,
            batch_norm=True, dilation_period=5, kernel_size=3, skip=True,
            subject_layers=True, n_subjects=len(config.subjects), subject_dim=0,
            complex_out=True, glu=2, glu_context=1, merger=True, initial_linear=270,
            gelu=True, merger_pos_dim=2048,
        )

    def forward(self, x, subjects):
        if subjects is None:
            raise ValueError('Missing subject index')
        batch = SimpleNamespace(subject_index=subjects,
                                positions=self.positions[None].expand(len(x), -1, -1))
        return F.normalize(self.encoder({'meg': x}, batch).mean(-1), dim=-1)


def build_model(config):
    if config.arch == 'eegnet':
        return EEGNet(config)
    if config.arch == 'brainmodule':
        return BrainModule(config)
    raise ValueError(f'Unknown baseline {config.arch}')
