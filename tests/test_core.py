import numpy as np
import pytest
import torch

from neuromoment.data import EventArchive, synthetic_archive
from neuromoment.losses import equivariance_ms, objective, shift_js, transport_anchor
from neuromoment.metrics import retrieval_metrics
from neuromoment.model import ModelConfig, NeuroMoment


torch.set_num_threads(1)


def test_anchor_direction_fractional_and_no_wrap():
    times = torch.arange(7, dtype=torch.float32)*0.1
    p = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])
    q, mask = transport_anchor(p, times, .1)
    assert q.argmax(-1).item() == 2  # Later anchor -> evidence earlier in window.
    q, _ = transport_anchor(p, times, .05)
    torch.testing.assert_close(q[0, 2:4], torch.tensor([.5, .5]), atol=1e-6, rtol=1e-5)
    edge, mask = transport_anchor(torch.tensor([[1., 0., 0., 0., 0., 0., 0.]]), times, .2)
    assert edge.sum() == 0 and not mask[0, -1]


def test_js_wasserstein_identity_and_gradients():
    times = torch.arange(8, dtype=torch.float32)*.1
    a = torch.tensor([[0., 0., .2, .5, .3, 0., 0., 0.]], requires_grad=True)
    b, _ = transport_anchor(a.detach(), times, .1)
    assert shift_js(a, b, times, .1).item() < 1e-6
    assert equivariance_ms(a, b, times, .1).item() < 1e-3
    shift_js(a, b, times, .1).backward()
    assert torch.isfinite(a.grad).all()
    with pytest.raises(ValueError, match="Empty"):
        shift_js(a, b, times, 2.)


@pytest.mark.parametrize('mode', ['adaptive', 'attention', 'fixed_prior', 'global_offset', 'fixed_window', 'mil'])
def test_modes_and_alignment_gradient(mode):
    torch.manual_seed(4)
    model = NeuroMoment(ModelConfig(mode=mode,sample_rate=100.,start=-.4,end=1.2,local_width=.3,moments=2,nms_interval=.3))
    out = model(torch.randn(4, 8, 160))
    assert out['alpha'].shape == (4, 7 if mode == 'fixed_window' else 40)
    torch.testing.assert_close(out['alpha'].sum(-1), torch.ones(4))
    loss = torch.nn.functional.cross_entropy(out['embedding'] @ torch.randn(32, 12), torch.arange(4))
    loss.backward()
    if mode == 'adaptive':
        grad = model.latency[-1].weight.grad
        assert grad is not None and grad.abs().sum() > 0
    if mode == 'global_offset':
        assert model.offset.grad.abs() > 0


def test_fine_branch_and_nms():
    model = NeuroMoment(ModelConfig(fine=True,fusion='all',sample_rate=100.,start=-.4,end=1.2,local_width=.3,moments=2,nms_interval=.3))
    out = model(torch.randn(3, 8, 160))
    pos = out['times'][out['indices']]
    assert ((pos[:, 1]-pos[:, 0]).abs() >= .3-1e-6).all()
    out['embedding'][:, 0].sum().backward()
    assert model.fine_encoder[0].weight.grad.abs().sum() > 0
    assert model.latency[-1].weight.grad.abs().sum() > 0  # Through soft branch.


def test_balanced_metrics_and_small_vocab():
    scores = torch.tensor([[9., 0., 0.], [9., 0., 0.], [9., 0., 0.], [9., 0., 0.]])
    m = retrieval_metrics(scores, torch.tensor([0, 0, 0, 1]))
    assert m['balanced_top1'] == .5
    assert m['balanced_top10'] == 1.
    assert m['coverage'] == 2/3
    assert m['chance_top10'] == 1.


def test_archive_real_recrop_and_margins(tmp_path):
    path = tmp_path/'fixture.npz'
    synthetic_archive(path)
    ds = EventArchive(path, 'train')
    x = ds.x[:2]
    torch.testing.assert_close(ds.view(x, .1), x[:, :, 30:190])
    with pytest.raises(ValueError, match='margins'):
        EventArchive(path, 'test', max_shift=.3)
    with pytest.raises(ValueError, match='exceeds'):
        ds.view(x, .5)


def test_two_view_objective(tmp_path):
    path = tmp_path/'fixture.npz'; synthetic_archive(path)
    ds = EventArchive(path, 'train')
    model = NeuroMoment(ModelConfig())
    a = model(ds.view(ds.x[:4])); b = model(ds.view(ds.x[:4], .1))
    loss, _ = objective(a, b, ds.y[:4], ds.embeddings, .1)
    loss.backward()
    assert torch.isfinite(loss)


def test_fixed_window_cannot_observe_outside_samples():
    model = NeuroMoment(ModelConfig(mode='fixed_window',sample_rate=100.,start=-.4,end=1.2,local_width=.3)).eval()
    x = torch.randn(2, 8, 160)
    modified = x.clone()
    modified[:, :, :50] += 100
    modified[:, :, 90:] -= 100
    # Fixed center .3 seconds, width .3 -> indices [55,85) for start=-.4.
    torch.testing.assert_close(model(x)['embedding'], model(modified)['embedding'])
