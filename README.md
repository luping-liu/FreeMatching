# FreeMatching

**Beyond Spatio-Temporal Priors: A Generalizable Approach for Dense Correspondence Matching**

Luping Liu, Bingyi Kang, Yifan Wang, Dong Xu

FreeMatching extends dense correspondence to image editing and reference-guided generation through generative and semantic foundation representations, heterogeneous correspondence supervision, and teacher-guided iterative refinement.

[Project](https://github.com/luping-liu/FreeMatching) · [Model weights](https://huggingface.co/luping-liu/FreeMatching)

## Installation

Extract the release ZIP and run the commands below from its `FreeMatching-release/` directory. For a Git clone, use the repository root containing `pyproject.toml`.

Use a fresh Python 3.10+ environment (packaging checks used Python 3.12). Install a CUDA-enabled PyTorch/torchvision pair for your hardware, then:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

This release vendors its modified Diffusers FLUX modules. No modifications to installed packages are needed. Obtain access to the official [DINOv3 model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), accept its terms, and run `hf auth login` with your own credentials. The base [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) and DINOv3 encoder are downloaded on first use or can be supplied as local directories.

## Inference

For inference only, no training data or teacher model is needed. Replace `source.jpg` and `target.jpg` below with your image paths:

```bash
python scripts/infer.py --source source.jpg --target target.jpg \
  --stage stage2 --output outputs/example
```

Stage 2 is the released EMA-refined model; select `--stage stage1` for the supervised model. The CLI downloads the chosen FreeMatching checkpoint automatically. To download both stages in advance and verify their SHA-256 hashes:

```bash
python scripts/download_checkpoints.py --stage both
python scripts/infer.py --source source.jpg --target target.jpg \
  --checkpoint checkpoints/stage2/model.safetensors --timestep 620
```

Images are resized to 512×288 by default. The model predicts **target-to-source** correspondence: output position `(x,y)` in the target identifies a sampling location in the source. Results include:

- `warped.png`: source sampled onto the target grid.
- `confidence.png` / `confidence.npy`: predicted covisibility confidence in `[0,1]`.
- `correspondence.npy`: `[2,H,W]` absolute normalized source coordinates, `(x,y)` order, `align_corners=True`.
- `flow.npy`: `[2,H,W]` target-to-source pixel displacement at the inference resolution.
- Resized input images and `metadata.json` with coordinate convention and timestep.

The default timestep is **620**, matching the original checkpoint evaluation, and is adjustable with `--timestep`. Height and width must be multiples of 16. For offline inference, provide `--diffusion-model /path/to/FLUX` and `--representation-model /path/to/DINOv3` together with a local checkpoint. BF16 CUDA is the default; CPU execution uses `--device cpu --dtype float32` and is memory-intensive.

## Repository layout

```text
freematching/             # Importable model and training components
  model.py               # Inference API
  flow_decoder.py        # Correspondence decoder
  datasets/              # Supervised and weak-pair dataset loaders
  training/              # Losses, teacher, sampling, and training helpers
  vendor/                # Modified Diffusers modules
scripts/
  infer.py
  download_checkpoints.py
  check_data.py
  prepare_data/          # Download, rendering, preprocessing, and conversion tools
configs/                 # Dataset paths and sampling configuration
docs/                    # Training, data preparation, and checkpoint guides
tests/
train_flow_flux.py        # Stage 1 training entry point
train_flow_flux_roma.py   # Stage 2 refinement entry point
```

`data/datasets/` is the default location for downloaded/generated data and is created during preparation; it is not part of the source release.

## Training and data

See [training instructions](docs/TRAINING.md), [data preparation](docs/DATA_PREPARATION.md), and [checkpoint details](docs/CHECKPOINTS.md). The release includes the nine supervised dataset loaders, weak image-pair loaders, Blender rendering, ScanNet++ preprocessing, and video trajectory conversion helpers.

To train, follow this order:

1. Install training dependencies: `pip install -r requirements-training.txt`.
2. Prepare the desired datasets using the data guide, edit `configs/datasets.json`, then run `python scripts/check_data.py --config configs/datasets.json`. This checks dataset roots; it does not validate all samples.
3. Run the S1 command in the training guide for supervised training, or download the released S1 weights with `python scripts/download_checkpoints.py --stage stage1` to start directly from S2.
4. Prepare weak image pairs, then run the S2 command in the training guide. Its default example requires filtered OmniGen2 manifests; the guide also explains UNO-only and custom-pair alternatives.

**Video data availability:** our tracked-video data consists of 520k clips from the ViPE dataset. Due to data size and source-video copyright restrictions, we do not distribute the videos or precomputed video trajectories. We provide loading/conversion code and preparation guidance for videos that users obtain and are authorized to use.

## Validation

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

Packaging checks cover tensor names/shapes against both full checkpoints, CPU tests of coordinate direction and warping, a small custom-transformer forward pass, trajectory-format conversion, and command-line imports. A full GPU inference/training run has not yet been performed on this packaged version.

## License

FreeMatching-authored code is licensed under [Apache-2.0](LICENSE). Third-party source notices are preserved in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); foundation models and datasets retain their respective terms.
