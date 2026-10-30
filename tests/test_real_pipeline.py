import csv
import json
from pathlib import Path
import wave

import numpy as np
import pytest
import torch

from neuromoment.bids import catalog, textgrid_words
from neuromoment.data import EventArchive, archive_fingerprint, synthetic_archive
from neuromoment.model import ModelConfig, NeuroMoment
from neuromoment.preprocess import build
from neuromoment.train import evaluate, main as train_main


def table(path, rows, delimiter='\t'):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter=delimiter)
        writer.writeheader();writer.writerows(rows)


def recording(header, data, sfreq=200, names=None):
    header.parent.mkdir(parents=True,exist_ok=True)
    names=names or [f'A{i+1}' for i in range(data.shape[0])]
    np.asarray(data.T,dtype='<f4').tofile(header.with_suffix('.eeg'))
    header.with_suffix('.vmrk').write_text('Brain Vision Data Exchange Marker File, Version 1.0\n[Common Infos]\nDataFile='+header.with_suffix('.eeg').name+'\n[Marker Infos]\n')
    header.write_text('Brain Vision Data Exchange Header File Version 1.0\n[Common Infos]\nCodepage=UTF-8\n'
        f'DataFile={header.with_suffix(".eeg").name}\nMarkerFile={header.with_suffix(".vmrk").name}\n'
        f'DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels={len(names)}\nSamplingInterval={1e6/sfreq}\n'
        '[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n[Channel Infos]\n'+
        '\n'.join(f'Ch{i+1}={ch},,1,V' for i,ch in enumerate(names)))
    table(header.with_name(header.name.replace('_eeg.vhdr','_channels.tsv')),
          [dict(name=n,type='EEG',status='good') for n in names])


def wav(path, seconds=14):
    path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as f:
        f.setnchannels(1);f.setsampwidth(2);f.setframerate(1000);f.writeframes(b'\0\0'*int(seconds*1000))


def ds_fixture(root):
    root.mkdir();(root/'dataset_description.json').write_text('{"Version":"test"}')
    labels=['THE','CAT']*4
    for run in range(1,4):
        wav(root/'stimuli'/f'audio{run:02d}.wav')
        intervals='\n'.join(f'intervals [{j+1}]:\nxmin = {j+3}\nxmax = {j+3.4}\ntext = "{w}"' for j,w in enumerate(labels))
        (root/'stimuli'/f'audio{run:02d}.TextGrid').write_text(
            'File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0\nxmax = 14\nitem []:\n'
            'item [1]:\nclass = "IntervalTier"\nname = "words"\nxmin = 0\nxmax = 14\nintervals: size = 8\n'+intervals)
        for subject in ['001','002']:
            t=np.arange(2800)/200
            x=np.stack([np.sin(2*np.pi*(7+i)*t+int(subject))*(i+1)*1e-5 for i in range(4)])
            header=root/f'sub-{subject}/eeg'/f'sub-{subject}_task-listening_run-{run:02d}_eeg.vhdr'
            recording(header,x)
    emb=root/'text.npz';np.savez(emb,words=np.array(['cat','the']),embeddings=np.eye(2,dtype='float32'))
    return emb


def test_textgrid_and_train_only_lazy_pipeline(tmp_path):
    root=tmp_path/'source';emb=ds_fixture(root)
    duration,events,_=textgrid_words(root/'stimuli/audio01.TextGrid')
    assert duration==14 and len(events)==8 and events[0]['word']=='the'
    out=tmp_path/'built'
    spec=build(root,'ds004408',emb,out,candidates=2,workers=1)
    train,val,test=[EventArchive(out,s) for s in ['train','val','test']]
    assert train.channels==4 and len(train)==16 and len(test)==16
    x,y=train[0]
    assert x.shape==(4,174) and torch.isfinite(x).all()
    assert len(train._cache)==1 and len(train.__getstate__()['_cache'])==0
    assert len(set(train.groups)&set(test.groups))==0
    # Only held-out source data change. Training calibration must remain exactly unchanged.
    for rec in spec['recordings']:
        if rec['split']=='train':continue
        p=Path(rec['source_header']).with_suffix('.eeg')
        data=np.fromfile(p,dtype='<f4')*100
        data.tofile(p)
    revised=build(root,'ds004408',emb,tmp_path/'revised',candidates=2,workers=1)
    assert spec['normalization']==revised['normalization']
    old=archive_fingerprint(out)
    rec=spec['recordings'][0];p=out/rec['path']
    arr=np.load(p);arr[0,0]+=1;np.save(p,arr)
    with pytest.raises(ValueError,match='Corrupted recording hash'):
        EventArchive(out,'train')
    # Dataset identity includes the descriptor, not just candidate words.
    descriptor=json.loads((out/'dataset.json').read_text());descriptor['guard']=2.1
    (out/'dataset.json').write_text(json.dumps(descriptor))
    assert archive_fingerprint(out)!=old


def test_brennan_clock_quarantine(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'dataset_description.json').write_text('{"Version":"test"}')
    table(root/'participants.tsv',[
        dict(participant_id='sub-001',analyzed='yes',exclusion_reason='n/a'),
        dict(participant_id='sub-002',analyzed='no',exclusion_reason='high_noise'),
        dict(participant_id='sub-003',analyzed='yes',exclusion_reason='n/a')])
    story=[]
    for segment in range(1,4):
        wav(root/'stimuli'/f'part{segment}.wav')
        for j in range(8):story.append(dict(Word=['the','cat'][j%2],Order=segment*100+j,Segment=segment,onset=j+3.))
    table(root/'sourcedata/AliceChapterOne-EEG.csv',story,',')
    for subject in ['001','002','003']:
        header=root/f'sub-{subject}/eeg'/f'sub-{subject}_task-alicelistening_eeg.vhdr'
        recording(header,np.ones((4,12000))*1e-5,names=['1','2','VEOG','AUD'])
        table(header.with_name(header.name.replace('_eeg.vhdr','_channels.tsv')),
              [dict(name=n,type=t,status='good') for n,t in [('1','EEG'),('2','EEG'),('VEOG','EOG'),('AUD','MISC')]])
        ev=[]
        for e in story:
            # Subject 3's annotations exceed raw bounds and must not be silently cropped.
            anchor=(e['Segment']-1)*20+(40 if subject=='003' else 0)
            ev.append(dict(word=e['Word'],word_index=e['Order'],segment=e['Segment'],
                           onset=anchor+e['onset'],duration=.3,stim_file=f"part{e['Segment']}.wav"))
        table(header.with_name(header.name.replace('_eeg.vhdr','_events.tsv')),ev)
    records,metadata=catalog(root,'brennan')
    assert metadata['channels']==['1','2']
    assert metadata['subjects']==['sub-001'] and len(records)==3
    assert {r['reason'] for r in metadata['excluded']}=={'high_noise','incompatible_source_timeline'}


def test_global_shuffled_prior_is_bounded_and_batch_independent(tmp_path):
    path=tmp_path/'data.npz';synthetic_archive(path)
    ds=EventArchive(path,'test');torch.manual_seed(4)
    model=NeuroMoment(ModelConfig()).eval()
    _,first=evaluate(model,ds,'cpu',batch_size=7,intervention='shuffled',intervention_seed=17)
    _,second=evaluate(model,ds,'cpu',batch_size=47,intervention='shuffled',intervention_seed=17)
    np.testing.assert_array_equal(first['prior_source_index'],second['prior_source_index'])
    assert (first['prior_source_index']!=np.arange(len(ds))).all()
    np.testing.assert_allclose(first['scores'],second['scores'],atol=1e-6,rtol=1e-5)


def test_epoch_resume_matches_uninterrupted_training(tmp_path,monkeypatch):
    path=tmp_path/'data.npz';synthetic_archive(path,classes=12)
    def run(out,epochs,resume=False):
        args=['train','--data',str(path),'--out',str(out),'--epochs',str(epochs),
              '--batch-size','64','--threads','1','--sampling','balanced','--log-every','0']
        if resume:args.append('--resume')
        monkeypatch.setattr('sys.argv',args);train_main()
    resumed=tmp_path/'resumed';full=tmp_path/'full'
    run(resumed,1);run(resumed,2,True);run(full,2)
    a=torch.load(resumed/'last.pt',weights_only=True);b=torch.load(full/'last.pt',weights_only=True)
    for key in a['model']:torch.testing.assert_close(a['model'][key],b['model'][key],atol=0,rtol=0)
    assert a['history'][-1]['val']==b['history'][-1]['val']


def test_eeg_control_and_output_lock(tmp_path):
    from neuromoment.controls import ShuffledEEG
    from neuromoment.train import output_lock
    path=tmp_path/'data.npz';synthetic_archive(path)
    ds=EventArchive(path,'test');control=ShuffledEEG(ds,17)
    assert (control.eeg_source_index!=np.arange(len(ds))).all()
    x,y=control[0];expected,_=ds[int(control.eeg_source_index[0])]
    torch.testing.assert_close(x,expected);assert y==ds.y[0]
    with output_lock(tmp_path/'run'):
        with pytest.raises(RuntimeError,match='already running'):
            with output_lock(tmp_path/'run'):pass
    with output_lock(tmp_path/'run'):pass


def test_filter_support_and_revision_gate(tmp_path):
    from neuromoment.preprocess import filter_spec
    from neuromoment.pipeline import run
    root=tmp_path/'source';emb=ds_fixture(root)
    assert 1 < filter_spec(512.00000001,100)[4] < 2
    with pytest.raises(ValueError,match='below finite preprocessing'):
        build(root,'ds004408',emb,tmp_path/'unsafe',candidates=2,guard=1.,workers=1)
    assert not (tmp_path/'unsafe/dataset.json').exists()
    out=tmp_path/'built';build(root,'ds004408',emb,out,candidates=2,workers=1)
    config={'root':str(root),'dataset':'ds004408','seed':7,'guard':2.,'sample_rate':128.,'candidates':2,
            'min_train':3,'audit':str(tmp_path/'audit.json'),'prepared':str(out),'embeddings':str(emb),
            'upstream_commit':'different-source'}
    p=tmp_path/'config.json';p.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='upstream revision'):
        run(p,'prepare')


def test_verifier_loads_compressed_index_once(tmp_path,monkeypatch):
    from neuromoment.verify import verify
    root=tmp_path/'source';emb=ds_fixture(root)
    out=tmp_path/'built';build(root,'ds004408',emb,out,candidates=2,workers=1)
    calls=[];original=np.lib.npyio.NpzFile.__getitem__
    def counted(self,key):
        calls.append(key);return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',counted)
    report=verify(out)
    assert report['recordings_sha256_verified']==6
    assert len(calls)<100  # Independent of the number of event rows; prevents quadratic decompression.


def test_absolute_time_metric_exposes_boundary_collapse():
    from neuromoment.diagnostics import absolute_equivariance_ms
    from neuromoment.train import training_objective
    times=torch.arange(7,dtype=torch.float32)*.1
    original=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
    shifted=torch.tensor([[0.,0.,1.,0.,0.,0.,0.]])
    assert absolute_equivariance_ms(original,shifted,times,.1).item()<1e-3
    boundary=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]])
    assert absolute_equivariance_ms(boundary,boundary,times,.2).item()==pytest.approx(200,abs=1e-3)
    embedding=torch.nn.functional.normalize(torch.randn(1,8),dim=-1).requires_grad_()
    out={'alpha':boundary,'embedding':embedding,'times':times}
    loss,_=training_objective(out,out,torch.tensor([0]),torch.randn(4,8),.2,lambda_shift=0,lambda_inv=0)
    loss.backward();assert torch.isfinite(loss) and embedding.grad.abs().sum()>0


def test_full_support_js_conserves_mass_and_penalizes_uniform_shift():
    from neuromoment.diagnostics import full_support_js,union_distributions
    times=torch.arange(7,dtype=torch.float32)*.1
    uniform=torch.ones(2,7)/7
    delta=torch.tensor([.15,-.15])
    p,q=union_distributions(uniform,uniform,times,delta)
    torch.testing.assert_close(p.sum(-1),torch.ones(2))
    torch.testing.assert_close(q.sum(-1),torch.ones(2),atol=1e-6,rtol=1e-6)
    assert full_support_js(uniform,uniform,times,delta).item()>0
    original=torch.tensor([[0.,0.,0.,1.,0.,0.,0.]])
    shifted=torch.tensor([[0.,0.,1.,0.,0.,0.,0.]])
    assert full_support_js(original,shifted,times,.1).item()<1e-6


def test_audit_respects_lower_training_frequency_threshold(tmp_path, monkeypatch):
    import neuromoment.audit as audit_module
    from neuromoment.protocol import assign_splits
    records = [{'stimulus': 'story', 'segment': str(i)} for i in range(3)]
    mapping = assign_splits([('story', str(i)) for i in range(3)])
    train_segment = next(segment for (_, segment), split in mapping.items() if split == 'train')
    events = [{'stimulus': 'story', 'segment': train_segment, 'word': word, 'event_id': i}
              for i, word in enumerate(['common', 'common', 'common', 'rare'])]
    monkeypatch.setattr(audit_module, 'catalog', lambda *args: (records, {}))
    monkeypatch.setattr(audit_module, 'usable_events', lambda *args: (events, 0))
    strict = audit_module.audit('unused', 'brennan', tmp_path/'strict.json', min_train=3)
    relaxed = audit_module.audit('unused', 'brennan', tmp_path/'relaxed.json', min_train=1)
    assert strict['candidate_pools']['250']['words'] == ['common']
    assert relaxed['candidate_pools']['250']['words'] == ['common', 'rare']
    assert relaxed['candidate_pools']['250']['min_train_independent_events'] == 1
