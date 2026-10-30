import json
from pathlib import Path

import numpy as np
import pytest
import torch

from neuromoment.baseline_models import BaselineConfig, build_model, SOURCE
from neuromoment.baseline_train import SubjectEvents, main as train_main
from neuromoment.data import EventArchive, synthetic_archive


@pytest.mark.parametrize('arch',['eegnet','brainmodule'])
def test_external_encoder_gradient_and_subject_metadata(arch):
    if arch == 'brainmodule' and not (SOURCE/'brainmagick_bm_models_common.py').exists():
        pytest.skip('Optional CC-BY-NC-4.0 sources not downloaded')
    torch.set_num_threads(1)
    c=BaselineConfig(arch,4,16,subjects=['s1','s2'],positions=[[0.,0.],[0.,1.],[1.,0.],[1.,1.]],brain_width=8,brain_depth=2)
    model=build_model(c)
    x=torch.randn(3,4,c.samples);subject=torch.tensor([0,1,0])
    prediction=model(x,subject)
    assert prediction.shape==(3,16)
    torch.testing.assert_close(prediction.norm(dim=-1),torch.ones(3),atol=1e-5,rtol=1e-5)
    prediction[:,0].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    if arch=='brainmodule':
        assert model.encoder.subject_layers.weights.grad.abs().sum()>0
        assert model.encoder.merger.heads.grad.abs().sum()>0
    else:
        with torch.no_grad():model.spatial.weight.mul_(100)
        model.constrain()
        assert model.spatial.weight.flatten(1).norm(dim=1).max()<=1.00001


def test_subject_shuffle_preserves_target_but_moves_source_identity(tmp_path):
    path=tmp_path/'data.npz';synthetic_archive(path,classes=4,channels=4)
    ds=EventArchive(path,'test');ds.subjects=np.array(['a','b']*(len(ds)//2))
    shuffled=SubjectEvents(ds,['a','b'],107)
    assert (shuffled.source_index!=np.arange(len(ds))).all()
    for i in range(len(ds)):
        source=int(shuffled.source_index[i]);x,y,s=shuffled[i]
        expected,_=ds[source]
        torch.testing.assert_close(x,expected)
        assert y==ds.y[i]
        assert s.item()==int(ds.subjects[source]=='b')


def test_eegnet_resume_includes_dropout_and_batchnorm(tmp_path,monkeypatch):
    path=tmp_path/'data.npz';synthetic_archive(path,classes=4,channels=4)
    def run(out,epochs,resume=False):
        args=['baseline_train','--arch','eegnet','--data',str(path),'--out',str(out),
              '--epochs',str(epochs),'--batch-size','16','--threads','1','--workers','0','--device','cpu','--skip-evaluation']
        if resume:args.append('--resume')
        monkeypatch.setattr('sys.argv',args);train_main()
    resumed=tmp_path/'resumed';full=tmp_path/'full'
    run(resumed,1);run(resumed,2,True);run(full,2)
    a=torch.load(resumed/'last.pt',weights_only=True);b=torch.load(full/'last.pt',weights_only=True)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)


def test_ridge_selection_does_not_change_when_only_test_labels_change(tmp_path,monkeypatch):
    from neuromoment.ridge import main as ridge_main
    path=tmp_path/'data.npz';synthetic_archive(path,classes=4,channels=4)
    def run(out):
        monkeypatch.setattr('sys.argv',['ridge','--data',str(path),'--out',str(out),
                                      '--lags','0','.3','--alphas','1','10','--threads','1'])
        ridge_main()
        return json.loads((out/'metrics.json').read_text())
    first=run(tmp_path/'first')
    with np.load(path,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    mask=arrays['split']=='test'
    arrays['y'][mask]=(arrays['y'][mask]+1)%4
    np.savez_compressed(path,**arrays)
    second=run(tmp_path/'second')
    assert (first['lag'],first['alpha'])==(second['lag'],second['alpha'])
    assert first['selection_split']==second['selection_split']=='val'
