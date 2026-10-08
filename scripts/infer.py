"""Run: python scripts/infer.py --source a.jpg --target b.jpg --stage stage2"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description='FreeMatching: warp source onto target')
    p.add_argument('--source', required=True)
    p.add_argument('--target', required=True)
    p.add_argument('--checkpoint', help='Local .safetensors or trusted legacy .pt checkpoint')
    p.add_argument('--stage', choices=['stage1', 'stage2'], default='stage2')
    p.add_argument('--output', default='outputs/example')
    p.add_argument('--diffusion-model', default=None, help='Official HF ID or local FLUX directory')
    p.add_argument('--representation-model', default=None, help='Official HF ID or local DINOv3 directory')
    p.add_argument('--timestep', type=int, default=620)
    p.add_argument('--height', type=int, default=288)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['bfloat16', 'float16', 'float32'], default='bfloat16')
    args = p.parse_args()
    import json
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from huggingface_hub import hf_hub_download
    from freematching.model import FreeMatching
    if args.height % 16 or args.width % 16 or min(args.height, args.width) <= 0:
        p.error('--height and --width must be positive multiples of 16')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        p.error('CUDA is unavailable; select --device cpu --dtype float32 for CPU execution')
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = hf_hub_download('luping-liu/FreeMatching', f'{args.stage}/model.safetensors')
        hf_hub_download('luping-liu/FreeMatching', f'{args.stage}/config.json')
    images = []
    for filename in [args.source, args.target]:
        with Image.open(filename) as image:
            rgb = ImageOps.exif_transpose(image).convert('RGB').resize((args.width, args.height), Image.Resampling.BILINEAR)
            images.append(torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float()[None] / 255)
    matcher = FreeMatching(checkpoint, diffusion_model=args.diffusion_model,
                           representation_model=args.representation_model,
                           device=args.device, dtype=getattr(torch, args.dtype), timestep=args.timestep)
    result = matcher(*images)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name in ['correspondence', 'confidence', 'flow']:
        np.save(out / f'{name}.npy', result[name][0].cpu().numpy())
    def save_image(tensor, name):
        array = (tensor[0].detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8')
        Image.fromarray(array.squeeze(2) if array.shape[2] == 1 else array).save(out / name)
    save_image(result['warped'], 'warped.png')
    save_image(result['confidence'], 'confidence.png')
    save_image(images[0], 'source.png'); save_image(images[1], 'target.png')
    (out / 'metadata.json').write_text(json.dumps({'direction':'target_to_source', 'height':args.height,
        'width':args.width, 'align_corners':True, 'timestep':args.timestep}, indent=2) + '\n')
    print(f'Saved outputs to {out.resolve()}')


if __name__ == '__main__':
    main()
