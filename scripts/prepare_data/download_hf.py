"""Download from official Hugging Face endpoints; no embedded credentials or mirrors."""
import argparse
from huggingface_hub import snapshot_download

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--include', nargs='+', default=None)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    snapshot_download(args.repo, repo_type='dataset', local_dir=args.output,
                      allow_patterns=args.include, max_workers=args.workers,
                      endpoint='https://huggingface.co')
