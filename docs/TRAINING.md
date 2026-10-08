# Training

The release includes supervised training (`train_flow_flux.py`) and teacher-guided iterative refinement (`train_flow_flux_roma.py`). Both use the project-local FLUX modules; do not overwrite Diffusers files in site-packages.

Install `requirements-training.txt`, prepare the datasets, and edit `configs/datasets.json`. DINOv3 and FLUX VAE/text-encoder weights are obtained separately from their official model repositories. The original two checkpoints are directly usable for inference; training is not required.

Run all commands from the release root after completing the README installation steps:

```bash
pip install -r requirements-training.txt
python scripts/check_data.py --config configs/datasets.json
```

## Stage 1

Example configuration for a new supervised run (8 GPUs, effective batch 32):

```bash
torchrun --standalone --nproc_per_node=8 train_flow_flux.py \
  --dataset_config configs/datasets.json \
  --input_mode double --sample_mode one --loss_type x0 \
  --mixed_precision bf16 --token_length 2 --timestep 620 \
  --img_height 288 --img_width 512 --batch_size 1 --accumulation_steps 4 \
  --output_decoder flow --decoder_warmup_steps 2000 \
  --learning_rate 3e-5 --gradient_checkpointing --use_ema --ema_decay 0.99 \
  --recipe_common_clean --max_train_steps_override 150000 \
  --output_dir outputs/stage1
```

Without `--init_ckpt_path`, initialization uses the foundation transformer and a new correspondence decoder. To continue from the released weights, pass `--init_ckpt_path checkpoints/stage1/model.safetensors`. This initializes weights only; `--resume` additionally restores the training state saved by a prior run in the selected output directory.

The command above is a public training recipe, not a claim of bit-for-bit reproduction of the historical checkpoint. The supplied S1 snapshot carries training counter 275000 from its original run. Rounded public data proportions, data regeneration, initialization, and sampling can affect a newly trained model.

## Stage 2

Download the released S1 initialization before running the example below:

```bash
python scripts/download_checkpoints.py --stage stage1
```

The released S2 weights are the EMA snapshot after 10000 refinement updates, initialized from the supplied S1 checkpoint. The selected original run freezes the output decoder and updates the transformer (including the semantic projection); the VAE, semantic encoder, and teacher remain frozen. EMA decay is 0.95. The original cycle has three supervised updates followed by one weak update; the launcher below retains that behavior.

```bash
torchrun --standalone --nproc_per_node=8 train_flow_flux_roma.py \
  --init_ckpt_path checkpoints/stage1/model.safetensors \
  --dataset_config configs/datasets.json \
  --use_roma_weak_sup --weak_data_sources uno,omnigen2 \
  --uno_label_root data/datasets/UNO-1M/labels \
  --omnigen2_root data/datasets/OmniGen2 \
  --mixed_precision bf16 --token_length 2 --timestep 620 \
  --batch_size 1 --accumulation_steps 4 --img_height 288 --img_width 512 \
  --sup_steps_per_cycle 3 --weak_steps_per_cycle 1 --ema_decay 0.95 \
  --sup_roma_loss_weight 0.0 --freeze_output_decoder \
  --loss_type x0 --output_decoder flow --learning_rate 3e-5 \
  --gradient_checkpointing --max_train_steps_override 10000 \
  --output_dir outputs/stage2
```

Prepare the specialized filtered OmniGen2 manifests before using this example; see [data preparation](DATA_PREPARATION.md). For a directly available UNO-only experiment, use `--weak_data_sources uno`. Generic user-owned image-pair manifests are also supported. These substitutions change the training data.

The current training implementation calls the official `romav2.RoMaV2` backend for fixed-teacher corrections. This backend name documents the implementation being distributed; it is distinct from naming the RoMa baseline in the paper. We retain the actual code path rather than substituting teacher architectures when packaging the trained weights. The teacher is not required at inference.

The training code retains its actual loss computation, including any gradient regularization. Default inference timestep is 620 to preserve checkpoint evaluation behavior; both training scripts accept `--timestep` explicitly.

## Runtime

CUDA is required for the distributed training examples. Use a CUDA-enabled PyTorch build and compatible GPU drivers. Optimizer states, the frozen teacher, and foundation encoders require substantially more memory than the 7.23-GiB checkpoint alone. Reduce per-device batch size or configure Accelerate/DeepSpeed for your hardware; no portable minimum-memory claim is made here.

W&B reporting is disabled by default. Enable it explicitly with `--report_to wandb` and authenticate using your own account. Checkpoints and debug images go to `--output_dir`; credentials, cluster addresses, and dataset payloads are not included in this release.
