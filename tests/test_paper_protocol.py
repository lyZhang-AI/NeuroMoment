"""Scientific-behavior checks for the settings specified in the supplied ISBI PDF."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import pytest
import torch

from neuromoment.model import ModelConfig, NeuroMoment, LocalTransformer
from neuromoment.protocol import assign_splits
from neuromoment.data import EventArchive, synthetic_archive
from neuromoment.losses import transport_anchor
from neuromoment.pipeline import finished
from neuromoment.train import main as train_main
from neuromoment.paper_statistics import paired_permutation, holm, compare, summarize

ROOT=Path(__file__).parents[1]


def test_paper_settings_and_five_distinct_run_splits():
    cfg=json.loads((ROOT/'configs/ds004408.json').read_text())
    repeated=json.loads((ROOT/'configs/paper.json').read_text())
    assert cfg['sample_rate']==128 and [cfg['low'],cfg['high']]==[1,30]
    assert [cfg['epoch_start'],cfg['epoch_end']]==[-.2,1.]
    assert cfg['text_model']=='google-t5/t5-large'
    assert cfg['training']['epochs']==100 and cfg['training']['batch_size']==64
    assert cfg['training']['lr']==3e-4 and cfg['training']['weight_decay']==.01
    assert len(repeated['split_seeds'])==5
    groups=[('story',f'run_{i:02d}') for i in range(20)]
    assert list(assign_splits(groups).values()).count('train')==16
    assert list(assign_splits(groups).values()).count('test')==2
    mappings=[assign_splits(groups,seed,cfg['split_counts']) for seed in repeated['split_seeds']]
    for m in mappings:assert [list(m.values()).count(k) for k in ['train','val','test']]==[16,2,2]
    assert len({tuple(sorted(m.items())) for m in mappings})==5
    with pytest.raises(ValueError,match='counts'):assign_splits(groups,7,[14,3,2])


def test_straight_through_local_path_gradient_and_deterministic_inference():
    torch.manual_seed(2)
    cfg=ModelConfig(channels=4,dim=8,text_dim=16,fine=True,fusion='local')
    model=NeuroMoment(cfg).train();x=torch.randn(2,4,154)
    training=model(x)
    assert training['indices'].shape==(2,3)
    assert training['alpha'].shape==(2,38)
    assert any(isinstance(m,LocalTransformer) for m in model.fine_encoder)
    training['embedding'][:,0].sum().backward()
    assert model.latency[-1].weight.grad.abs().sum()>0  # No soft/global fusion path exists here.
    assert model.fine_encoder[0].weight.grad.abs().sum()>0
    model.eval()
    first=model(x);second=model(x)
    torch.testing.assert_close(first['embedding'],second['embedding'],atol=0,rtol=0)
    torch.testing.assert_close(training['embedding'],first['embedding'],atol=1e-6,rtol=1e-5)
    times=first['times'][first['indices']].sort(-1).values
    assert ((times[:,1:]-times[:,:-1])>=.12).all()
    torch.testing.assert_close(model.delays,torch.arange(26)*.04,atol=1e-7,rtol=1e-6)
    # Negative control: hard-only local inference cannot train the latency predictor.
    cfg.selection='hard';hard=NeuroMoment(cfg).train();hard.load_state_dict(model.state_dict())
    hard(x)['embedding'][:,0].sum().backward()
    assert hard.latency[-1].weight.grad is None or hard.latency[-1].weight.grad.abs().sum()==0


def test_positive_signal_shift_moves_response_later_under_fixed_anchor():
    ds=EventArchive.__new__(EventArchive)
    ds.sample_rate=128.;ds.context_start=-.28;ds.start=-.2;ds.n=154
    context=torch.zeros(1,1,174);context[0,0,70]=1
    delta=10/128
    base=ds.view(context)
    shifted=ds.view(context,-delta)
    assert shifted.argmax(-1).item()-base.argmax(-1).item()==10
    times=torch.arange(154)/128-.2
    expected,_=transport_anchor(base[:,0],times,-delta)
    torch.testing.assert_close(expected,shifted[:,0],atol=2e-6,rtol=1e-5)


def test_paper_filter_and_continuous_context(tmp_path):
    # Reuse an actual BrainVision fixture and exercise the complete 128Hz preprocessing path.
    from test_real_pipeline import ds_fixture
    from neuromoment.preprocess import build, filter_spec
    root=tmp_path/'source';emb=ds_fixture(root)
    assert filter_spec(200,128,1,30)[0]['h_freq']==30
    out=tmp_path/'eeg'
    spec=build(root,'ds004408',emb,out,candidates=2,workers=1,sample_rate=128,
               split_counts=[1,1,1],low=1,high=30,context_start=-.28,context_end=1.08,epoch_start=-.2,epoch_end=1.)
    ds=EventArchive(out,'train',max_shift=.08)
    assert ds.context_shape==(4,174) and ds.n==154
    x,_=ds[0]
    assert ds.view(x[None],-.08).shape[-1]==154
    assert ds.view(x[None],.08).shape[-1]==154
    assert spec['preprocessing']['bandpass_hz']==[1,30]


def test_early_stopping_marker_and_resume(tmp_path,monkeypatch):
    data=tmp_path/'events.npz';synthetic_archive(data,classes=4,channels=4)
    out=tmp_path/'run'
    args=['train','--data',str(data),'--out',str(out),'--epochs','5','--patience','1',
          '--min-delta','100','--threads','1','--workers','0','--batch-size','64','--lambda-shift','0','--log-every','0']
    monkeypatch.setattr(sys,'argv',args);train_main()
    marker=json.loads((out/'training_complete.json').read_text())
    assert marker=={'epochs_requested':5,'epochs_completed':2,'early_stopped':True}
    assert finished(out,5)
    before=(out/'history.json').read_bytes()
    monkeypatch.setattr(sys,'argv',args+['--resume']);train_main()
    assert (out/'history.json').read_bytes()==before


def test_exact_two_sided_subject_permutation_and_holm():
    p,meta=paired_permutation(np.ones((4,3)))
    np.testing.assert_allclose(p,[.125]*3)
    assert meta['exact'] and meta['permutations']==16 and meta['subjects']==4
    np.testing.assert_allclose(holm([.01,.04,.02]),[.03,.04,.04])
    identical,_=paired_permutation(np.zeros((4,3)))
    np.testing.assert_array_equal(identical,[1,1,1])


def test_five_split_permutation_averages_subjects_before_testing(tmp_path):
    first=[];second=[]
    for split in range(5):
        metadata=dict(labels=np.array([0,1,0,1]),subjects=np.array(['a','b','c','d']),
                      words=np.array(['cat','dog']),groups=np.repeat(f'run_{split}',4),event_ids=np.array(['1','2','1','2']))
        a=tmp_path/f'a{split}.npz';b=tmp_path/f'b{split}.npz'
        np.savez(a,scores=np.array([[2,0],[0,2],[2,0],[0,2]]),**metadata)
        np.savez(b,scores=np.array([[0,2],[2,0],[0,2],[2,0]]),**metadata)
        first.append(a);second.append(b)
    result=compare(first,second,expected_subjects=4)
    assert result['test']['subjects']==4 and result['matched_splits']==5
    np.testing.assert_allclose(result['raw_p'],[1,1,.125])
    with pytest.raises(ValueError,match='five'):compare(first[:3],second[:3],expected_subjects=4)


def test_no_missing_experiments_are_filled_with_manuscript_numbers(tmp_path):
    result=summarize(tmp_path,[7,17,27,37,47],['neuromoment'])
    assert result['rows']==[] and len(result['missing_artifacts'])==5


def test_t5_exports_encoder_only_and_masked_mean(tmp_path,monkeypatch):
    from neuromoment.text import export
    local=tmp_path/'t5';local.mkdir();(local/'config.json').write_text('{"model_type":"t5"}')
    class Tokenizer:
        @staticmethod
        def from_pretrained(*a,**k):return Tokenizer()
        def __call__(self,words,**k):return {'input_ids':torch.ones(len(words),3,dtype=torch.long),'attention_mask':torch.tensor([[1,1,0]]*len(words))}
    class Encoder(torch.nn.Module):
        @staticmethod
        def from_pretrained(*a,**k):return Encoder()
        def forward(self,**inputs):
            values=torch.arange(3,dtype=torch.float32)[None,:,None].expand(len(inputs['input_ids']),3,1024)
            return types.SimpleNamespace(last_hidden_state=values)
    fake=types.ModuleType('transformers');fake.AutoTokenizer=Tokenizer;fake.T5EncoderModel=Encoder
    fake.AutoModel=types.SimpleNamespace(from_pretrained=lambda *a,**k:pytest.fail('T5 must use encoder only'))
    hf=types.ModuleType('huggingface_hub');hf.snapshot_download=lambda *a,**k:pytest.fail('Local test must not access network')
    monkeypatch.setitem(sys.modules,'transformers',fake);monkeypatch.setitem(sys.modules,'huggingface_hub',hf)
    out=tmp_path/'embeddings.npz';export(['cat','dog'],out,model_id=str(local))
    with np.load(out) as f:
        assert f['embeddings'].shape==(2,1024)
        np.testing.assert_allclose(np.linalg.norm(f['embeddings'],axis=1),1,atol=1e-6)
    assert json.loads(Path(str(out)+'.json').read_text())['frozen']


def test_three_moment_masking_matches_union_coverage_without_changing_selection():
    from neuromoment.occlusion import coverage_catalog,matched_masks
    model=NeuroMoment(ModelConfig(channels=4,dim=8,text_dim=16,fine=True,fusion='all')).eval()
    with torch.no_grad():out=model(torch.randn(8,4,154))
    catalog=coverage_catalog(model,154)
    selected,random=matched_masks(out['indices'].numpy(),catalog,np.random.default_rng(107))
    np.testing.assert_array_equal(selected.sum(1),random.sum(1))
    assert (selected!=random).any(1).all()
    repeat=matched_masks(out['indices'].numpy(),catalog,np.random.default_rng(107))
    np.testing.assert_array_equal(random,repeat[1])
    # Allow true overlap: 120ms exclusion is smaller than a 160ms crop.
    assert min(catalog[1])<max(catalog[1])


def test_repeated_runner_creates_independent_vocabularies_with_fixed_initialization():
    path=ROOT/'scripts/run_paper.py'
    spec=importlib.util.spec_from_file_location('paper_runner',path)
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    base=json.loads((ROOT/'configs/ds004408.json').read_text())
    configs=[runner.split_config(base,seed,7) for seed in [7,17,27,37,47]]
    assert len({c['prepared'] for c in configs})==5
    assert len({c['embeddings'] for c in configs})==5
    assert len({c['audit'] for c in configs})==5
    assert {c['training_seed'] for c in configs}=={7}
    assert {c['seed'] for c in configs}=={7,17,27,37,47}
