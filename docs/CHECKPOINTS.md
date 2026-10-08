# Checkpoints

Both public snapshots contain 257 tensors: the correspondence transformer (including DINO feature projection) and the flow decoder. They are exported to safetensors without optimizer state, experiment metadata, filesystem paths, or tracking-service state. The frozen FLUX VAE/text encoder, DINOv3 encoder, and fixed teacher are not bundled.

| Stage | Saved step | EMA | Bytes | SHA-256 |
| --- | ---: | --- | ---: | --- |
| stage1 | 275000 | false | 7,762,299,928 | `a2c0314c09a22fa344d1e7bead63ff469cfb21781269202ee0f7fbf509a3f4d1` |
| stage2 | 10000 | true | 7,762,299,928 | `6f28147a41a11b1691b76eeba108a8fd97fdd41e5ec72bcd83e020ed51753880` |

These saved-step counters identify the original snapshots. S2 is initialized from the released S1 and uses its EMA weights after 10k adaptation updates.

Both use FLUX.2-klein-base-4B, DINOv3 ViT-L/16, the learned flow decoder, 512×288 default input size, two text tokens, BF16 weights, and configurable timestep 620. Correspondence maps run from target pixels to source coordinates with `align_corners=True`.

Each stage has a companion `config.json`. Use `scripts/download_checkpoints.py` to download and verify files, or `scripts/infer.py` for automatic download and inference. The `.safetensors` files are weights, not standalone Transformers `AutoModel` repositories; load them with the included FreeMatching implementation.

Packaging validation compared every tensor name and shape with the release architecture. It does not replace a full GPU benchmark run; see the validation section in README.
