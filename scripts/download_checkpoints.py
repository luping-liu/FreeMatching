"""Download public tensor-only checkpoints and verify SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['stage1', 'stage2', 'both'], default='both')
    p.add_argument('--output', default='checkpoints')
    args = p.parse_args()
    from huggingface_hub import hf_hub_download
    for stage in ['stage1', 'stage2'] if args.stage == 'both' else [args.stage]:
        config = Path(hf_hub_download('luping-liu/FreeMatching', f'{stage}/config.json', local_dir=args.output))
        weights = Path(hf_hub_download('luping-liu/FreeMatching', f'{stage}/model.safetensors', local_dir=args.output))
        info = json.loads(config.read_text())
        with weights.open('rb') as f:
            hasher = hashlib.sha256()
            for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
                hasher.update(block)
            digest = hasher.hexdigest()
        if digest != info['sha256'] or weights.stat().st_size != info['size_bytes']:
            raise RuntimeError(f'{stage}: checkpoint integrity verification failed')
        print(f'{stage}: verified {weights}')


if __name__ == '__main__':
    main()
