"""Check dataset roots without launching training or indexing entire datasets."""
import argparse
import json
import os
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/datasets.json')
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    failed = False
    for name, spec in config.items():
        if spec['ratio'] <= 0:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(spec['path'])))
        good = path.is_dir() and next(path.iterdir(), None) is not None
        print(f'{name}: {"OK" if good else "MISSING/EMPTY"} ({path})')
        failed |= not good
    raise SystemExit(1 if failed else 0)
