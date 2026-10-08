# Data preparation

Run commands from the release root. Downloads use official project sites or the public Hugging Face repositories linked below. No training data is bundled. Obtain access to gated datasets under their respective terms before downloading them; authenticate with `hf auth login` when required.

`configs/datasets.json` contains editable paths and sampling proportions. Its nine-source recipe uses ScanNet++ 15%, BlendedMVS 10%, FlyingThings3D 10%, Dynamic Replica 10%, Spring 5%, PointOdyssey 5%, FlyingChairs 5%, Video 20%, and Blender 20%. These are the rounded proportions in the paper. Set unavailable datasets to ratio 0 to run a smaller experiment; the sampler renormalizes active sources. Such experiments use a different mixture.

The default root is `data/datasets`. To change all built-in paths, set `FREEMATCHING_DATA_ROOT` and omit `--dataset_config`; when passing a JSON config, edit its paths directly (environment variables in those paths are expanded).

## BlendedMVS

Source: [BlendedMVS](https://github.com/YoYo000/BlendedMVS); compatible processed images/depth/cameras: [UFM public mirror](https://huggingface.co/datasets/infinity1096/blendedmvs_processed).

```bash
python scripts/prepare_data/download_hf.py --repo infinity1096/blendedmvs_processed \
  --output data/datasets/BlendedMVS/archives --include '*.zip'
mkdir -p data/datasets/BlendedMVS/blendedmvs_processed_ufm
for archive in data/datasets/BlendedMVS/archives/*.zip; do
  unzip -n "$archive" -d data/datasets/BlendedMVS/blendedmvs_processed_ufm
done
curl --fail --location --retry 5 \
  https://download.europe.naverlabs.com/ComputerVision/DUSt3R/blendedmvs_pairs.npy \
  --output data/datasets/BlendedMVS/blendedmvs_pairs.npy
```

The pair metadata is supplied separately by [DUSt3R](https://github.com/naver/dust3r). Keep its name/location as shown: the loader derives this path from `blendedmvs_processed_ufm`. Scene directories must be immediately inside that folder. The loader computes correspondence from depth and camera poses and builds its local index on first use.

## ScanNet++

Request access through the [official ScanNet++ site](https://kaldir.vc.in.tum.de/scannetpp/), then use its downloader with your own credentials to obtain DSLR imagery, calibration, and scans. Expected raw layout: `data/datasets/scannet++/raw/data/<scene>/{dslr,scans}`.

```bash
python scripts/prepare_data/download_hf.py --repo infinity1096/ufm_metadata \
  --output data/datasets/scannet++/ufm_metadata --include 'snpp_aggregated_metadata_train.npz'
pip install trimesh scipy opencv-python-headless
pip install git+https://github.com/NVlabs/nvdiffrast.git
python scripts/prepare_data/preprocess_scannetpp_cuda.py \
  --scannetpp_dir data/datasets/scannet++/raw \
  --precomputed_anymap_pairs data/datasets/scannet++/ufm_metadata/snpp_aggregated_metadata_train.npz \
  --output_dir data/datasets/scannet++/preprocess_scannetpp_ufm/train
python scripts/prepare_data/build_scannetpp_mmap_cache.py \
  --ufm_output_dir data/datasets/scannet++/preprocess_scannetpp_ufm/train
```

The included renderer uses CUDA rasterization and does not require a display server. For multi-GPU preprocessing use separate processes with `--world_size N --rank i`, each bound to a GPU. This renderer requires a CUDA toolchain; it has not been GPU-tested as part of this release packaging pass.

## FlyingThings3D

Use the [Scene Flow dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) or the compatible [UFM HDF5 mirror](https://huggingface.co/datasets/infinity1096/flyingthings3d_processed):

```bash
python scripts/prepare_data/download_hf.py --repo infinity1096/flyingthings3d_processed \
  --output data/datasets/flyingthings/flyingthings_preprocessed_ufm
```

Preserve the repository directory structure, including `individual_quantities/train/index.json`, `flow/train/job_df.csv`, and all referenced HDF5 shards. The HDF5 reader is included in `freematching/datasets/distributed_h5.py`.

## Spring

Download the training images, bidirectional flow, masks, and camera data from [Spring](https://spring-benchmark.org/) / [DaRUS](https://darus.uni-stuttgart.de/). The preparation workflow uses the [UFM training branch](https://github.com/UniFlowMatch/UFM/tree/train):

```bash
git clone --branch train https://github.com/UniFlowMatch/UFM.git external/UFM
# Install the preprocessing dependencies documented by UFM.
python external/UFM/preprocess_datasets/preprocess_spring.py \
  --spring_root data/datasets/spring/raw/spring \
  --output_dir data/datasets/spring/spring_processed_ufm
```

Retain the processed directory hierarchy and flow/mask files; the loader generates its index on first use. The rebuilding workflow used these DaRUS file IDs: left/right frames `199097/199098`, forward flows `199011/242150`, backward flows `199006/199103`, maps `242149`, and camera data `198954`. Official file URLs have the form `https://darus.uni-stuttgart.de/api/access/datafile/ID`.

## PointOdyssey

Download [official PointOdyssey files](https://huggingface.co/datasets/aharley/pointodyssey). Concatenate multipart archives in filename order before extraction.

```bash
python scripts/prepare_data/download_hf.py --repo aharley/pointodyssey \
  --output data/datasets/pointodyssey/archives --include 'train.tar.gz.part*'
cat data/datasets/pointodyssey/archives/train.tar.gz.part* > data/datasets/pointodyssey/archives/train.tar.gz
tar -xzf data/datasets/pointodyssey/archives/train.tar.gz -C data/datasets/pointodyssey
python -m freematching.datasets.pointodyssey --task preprocess \
  --data_root data/datasets/pointodyssey --s_frames 20 --num_workers 8
```

Check that sequences are at `pointodyssey/train/<sequence>/`; move any extra archive wrapper directory accordingly. The preprocessing step writes trajectory/visibility memmaps and local metadata. Regenerate caches after moving the dataset to a different location.

## Dynamic Replica

Follow [DynamicStereo's official dataset instructions](https://github.com/facebookresearch/dynamic_stereo). Download the training sequence images, optical flow/visibility, and trajectories; place extracted sequences under `data/datasets/DynamicStereo/train` with their original hierarchy. The sparse builder selects tracked correspondences and caches the sequence index on first use. The public release does not ship workstation-specific index caches.

## FlyingChairs

Download the original [FlyingChairs archive and train/validation split file](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html). Extract to `data/datasets/flyingchairs`, retaining the `FlyingChairs_release/data` folder and `FlyingChairs_train_val.txt` required by torchvision's FlyingChairs loader.

## Blender / Objaverse

Obtain object meshes from [Objaverse](https://huggingface.co/datasets/allenai/objaverse), retaining the license associated with each asset. Install Blender with Cycles support, then run the included renderer on a directory containing GLB files:

```bash
blender -b -P scripts/prepare_data/render_blender_pairs.py -- \
  data/datasets/objaverse/meshes data/datasets/objaverse/render_output_all 1000 0 0
```

The final three arguments are the number of pairs, GPU index, and starting sample index. Different workers must use nonoverlapping sample indices. Outputs consumed by the loader are:

```text
render_output_all/
  frames/sample_00000_0001.webp
  frames/sample_00000_0002.webp
  masks/...
  flow_data/...
```

Each sample has two rendered frames, masks, and EXR correspondence data. Install `OpenEXR` for the loader. The paper's Blender count is in **image pairs**. The renderer is provided for regeneration, not as a download of the original rendered set.

## Tracked video

Due to the size of the video data and copyright restrictions on the source videos, we do not distribute the video corpus or its precomputed trajectory annotations. We provide the data loader, trajectory-format conversion code, and preparation instructions for videos that users obtain and are authorized to use. Our tracked-video data consists of **520k clips from the ViPE dataset** ([ViPE project](https://research.nvidia.com/labs/toronto-ai/vipe/)). We generate trajectory annotations with AllTracker and sample frame pairs from these clips during loading.

Use [AllTracker](https://github.com/aharley/alltracker) to track from the first frame forward and the last frame backward, sampling every 8 frames in each direction. Preserve the full-rate source video. From its `forward_sliding` output, add the pixel grid to displacement to obtain absolute `(x,y)` trajectories. Save an NPZ containing `forward_tracks`, `backward_tracks` of shape `[T,2,H,W]`, and `forward_confidence`, `backward_confidence` of shape `[T,1,H,W]`, in `[0,1]`. The two directions must be generated independently; backward index 0 refers to the original last frame. Use trajectories at the original video pixel resolution, or resize both the video and trajectories consistently before export.

```bash
python scripts/prepare_data/export_video_tracks.py --tracks tracks.npz \
  --output data/datasets/vipe/example__supp
```

Place `example.mp4` beside `example__supp/`. The exporter produces `metadata.json` plus `fwd-000.tiff...` and `bwd-000.tiff...`, each storing x, y, and confidence pages. At least five tracked frames per direction are required. The loader defaults to frame stride 8; changing the tracking stride requires changing its `downsample_ratio` consistently. Automatic end-to-end video annotation is not bundled; the official tracker runs separately and its outputs are converted with the included exporter.

## Weak image pairs (Stage 2)

[UNO-1M](https://huggingface.co/datasets/bytedance-research/UNO-1M): download `labels/split*.json` and image archives. Extract the archives under `data/datasets/UNO-1M/images_full`, preserving the paths referenced by labels. Supply `--uno_label_root data/datasets/UNO-1M/labels`. The loader applies its score filter and forms image pairs without correspondence labels.

[OmniGen2 X2I2](https://huggingface.co/datasets/OmniGen2/X2I2): download the desired video editing/generation JSONL files and their corresponding image archives. Multipart tar files must be concatenated before extraction. Preserve relative image paths under `data/datasets/OmniGen2/images`.

The specialized OmniGen2 loader accepts `merged_shards_video/merged_part_*.jsonl`, with fields `source`, `reference_images`, `target_image`, and `match_ratio_im_B`. Its filtering expects quality annotations computed during preprocessing; upstream X2I2 JSONL files are not automatically equivalent to these filtered manifests. Do not invent missing quality scores. For new data without those annotations, use the generic image-only manifest interface instead:

```json
[{"source_image": "source.jpg", "target_image": "target.jpg"}]
```

Pass `--weak_data_sources matching --matching_weak_manifest pairs.json --matching_image_root /path/to/images`. This supports user-prepared weak pairs directly; it does not reproduce the historical filtered pair selection.

## Before training

```bash
python scripts/check_data.py --config configs/datasets.json
```

This checks nonempty roots, not the full contents of large datasets. Start with a small run after preparing each source. Generated local caches may embed your chosen filesystem paths and should not be redistributed.
