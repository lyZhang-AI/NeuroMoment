"""Keep run partitions distinct from optimization-seed repetitions."""
import importlib.util
import json
from pathlib import Path
from neuromoment.paper_statistics import summarize

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('paper_queue',ROOT/'scripts/run_paper.py')
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


def test_training_seed_does_not_rebuild_or_change_split():
    base=json.loads((ROOT/'configs/ds004408.json').read_text())
    first=queue.split_config(base,7,7);second=queue.split_config(base,7,17)
    assert first['seed']==second['seed']==7
    for key in ['prepared','embeddings','audit','split_counts']:assert first[key]==second[key]
    assert first['training_seed']!=second['training_seed']


def test_incomplete_queue_does_not_invent_results(tmp_path):
    report=summarize(tmp_path,[7,17,27,37,47],['neuromoment'])
    assert report['rows']==[] and len(report['missing_artifacts'])==5
    assert report['split_seeds']==[7,17,27,37,47]
