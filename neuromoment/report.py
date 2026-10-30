"""Generate five-split paper reports from completed evaluation artifacts."""
import argparse
import json
from pathlib import Path
from .paper_statistics import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default='configs/paper.json')
    p.add_argument('--models',nargs='+')
    a=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    spec=json.loads((project/a.config).read_text())
    base=json.loads((project/spec['base_config']).read_text())
    result=summarize(project,spec['split_seeds'],a.models or [m['name'] for m in base['models']])
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
