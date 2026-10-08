# Third-party components

FreeMatching-authored code is distributed under Apache-2.0 (see LICENSE). This does not replace the licenses of dependencies, pretrained models, or datasets.

- `freematching/vendor/model_flux2.py` and `pipeline_flux2_klein.py` are modified from Hugging Face Diffusers / Black Forest Labs under Apache-2.0. Original copyright headers are retained. The local modifications add DINO feature injection and parallel image latent preparation/decoding.
- The dataset loaders and preprocessing helpers build on [UFM](https://github.com/UniFlowMatch/UFM/tree/train). Its BSD-3-Clause license is included in `licenses/UFM-LICENSE`. In particular, the HDF5 reader and ScanNet++ preprocessing follow the UFM implementation; the latter adds a CUDA rasterization backend.
- The optional TinySD3 decoder and common decoder blocks follow [TAESD](https://github.com/madebyollin/taesd). Its MIT license is included in `licenses/TAESD-LICENSE`.
- [FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) is obtained separately under its upstream Apache-2.0 license.
- [DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) is obtained separately and is governed by the DINOv3 license and access conditions. Its pretrained encoder is not included in the FreeMatching checkpoints.
- Stage-2 training imports [RoMaV2](https://github.com/Parskatt/RoMaV2) from its official package (MIT, with separately licensed DINOv3 components). Teacher weights are not included here.
- [AllTracker](https://github.com/aharley/alltracker), [nvdiffrast](https://github.com/NVlabs/nvdiffrast), Blender, and the individual datasets retain their upstream licenses. They are installed/downloaded separately.
