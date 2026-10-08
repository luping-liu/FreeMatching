"""
https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""

import os
import re
import sys
import math
import json
import shutil
import inspect
import copy
import random
import itertools
import argparse
import numpy as np
import wandb.util
from tqdm.auto import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange
from safetensors.torch import load_model
from pathlib import Path 

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.distributed as dist
from datetime import timedelta
from torch.utils.data import Dataset
from torchvision import transforms
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from freematching.vendor.model_flux2 import Flux2Transformer2DModel
from freematching.vendor.pipeline_flux2_klein import Flux2KleinPipeline
from freematching.checkpoint import load_checkpoint
from freematching.datasets.unified_loader import load_dataset_config
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel 
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from freematching.datasets.unified_loader import DEFAULT_DATASET_CONFIGS_7, UnifiedDataloaderBuilder
from freematching.training.losses import flow_gradient_loss
from freematching.training.utils import debug_warping, expand_mask_boundary
from freematching.flow_decoder import FluxFlowModel, build_output_decoder

MASK_GT_SKIP_DATASETS = {"tracethings", "pointodyssey", "dynamic_replica", "flyingchairs"}
MASK_GT_SKIP_BENCHMARKS = {"kitti", "eth3d_dense"}

# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="FreeMatching supervised training")
    parser.add_argument("--diffusion_model", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--repre_model", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")

    parser.add_argument("--output_dir", type=str, default="./output", help="模型预测和检查点的输出目录。")
    parser.add_argument("--ckpt_dir", type=str, default="", help="用于恢复训练的特定检查点目录。")
    parser.add_argument(
        "--init_ckpt_path",
        type=str,
        default="",
        help="Initialization checkpoint; supports raw vis or {vis,flow_decoder} format.",
    )

    parser.add_argument("--input_mode", type=str, default="double", choices=["single", "double"], help="选择输入数据流模式: 'single' (A,B,Flow合并) 或 'double' (A,B作为条件, Flow为目标)。")
    parser.add_argument("--sample_mode", type=str, default="one", choices=["multi", "one"], help="数据集采样模式。")
    parser.add_argument("--data_refine", action="store_true", ) 

    # --- 训练配置 ---
    parser.add_argument("--batch_size", type=int, default=1, help="训练数据加载器的批次大小（单设备）。")
    parser.add_argument("--num_samples_per_epoch", type=int, default=None, help="每个 epoch 采样的样本数。如果为 None, 则使用整个数据集的长度。")
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps_override", type=int, default=None)
    parser.add_argument("--recipe_common_clean", action="store_true")
    parser.add_argument("--dataset_sources", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--use_ema", action="store_true", help="是否额外维护并保存 EMA 模型。")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay，仅在 --use_ema 时生效。")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="是否使用混合精度。")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="是否使用梯度检查点以节省内存。")
    
    # --- 模型与损失配置 ---
    parser.add_argument("--token_length", type=int, default=2)
    # parser.add_argument("--resolution", type=int, default=512, help="Legacy square-resolution argument; current loader uses --img_height/--img_width.")
    parser.add_argument("--img_height", type=int, default=288, help="Training image height. Must be divisible by 16 for FLUX latent packing.")
    parser.add_argument("--img_width", type=int, default=512, help="Training image width. Must be divisible by 16 for FLUX latent packing.")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--loss_type", type=str, default="x0", choices=["eps", "x0", "vlc"])
    parser.add_argument("--output_decoder", type=str, default="flow", choices=["vae", "tiny_sd3", "flow"],
                        help="输出侧 decoder: 'vae' 保持原始 FLUX2 VAE decode, 其他选项使用可训练 32-channel latent decoder。")
    parser.add_argument("--taesd_decoder_ckpt", type=str, default="",
                        help="可选 TinySD3Decoder 权重路径，仅在 --output_decoder tiny_sd3 时加载。")
    parser.add_argument("--decoder_warmup_steps", type=int, default=2000,
                        help="使用可训练输出 decoder 时，前 N 个 optimizer step 冻结 DiT，仅训练 decoder。")
    
    # --- 日志与保存 ---
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--max_saved_models", type=int, default=5, help="最多保留的模型权重检查点数量。")

    
    # --- 系统 ---
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume", action="store_true", help="从 output_dir 中的最新检查点恢复。")
    parser.add_argument("--debug", action="store_true", help="启用调试模式以进行快速测试。")
    
    parser.add_argument("--timestep", type=int, default=620)
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--report_to", choices=["none", "wandb"], default="none")
    args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args


@torch.no_grad()
def update_ema_model(ema_model, source_model, decay):
    for ema_param, src_param in zip(ema_model.parameters(), source_model.parameters()):
        ema_param.data.mul_(decay).add_(src_param.data.to(ema_param.dtype), alpha=1.0 - decay)
    for ema_buffer, src_buffer in zip(ema_model.buffers(), source_model.buffers()):
        ema_buffer.data.copy_(src_buffer.data.to(ema_buffer.dtype))


def strip_unused_state_dict_keys(state_dict):
    return {key: value for key, value in state_dict.items() if not key.startswith("dpt_head.")}


def set_vis_trainable_for_decoder_warmup(model, enabled: bool):
    flow_decoder = getattr(model, "output_decoder", None)
    if flow_decoder is None:
        return True
    return enabled


def split_wrapped_state_dict(state_dict):
    vis_state = {}
    decoder_state = {}
    for key, value in state_dict.items():
        if key.startswith("vis."):
            vis_state[key[len("vis."):]] = value
        elif key.startswith("output_decoder."):
            decoder_state[key[len("output_decoder."):]] = value
    return vis_state, decoder_state


def load_checkpoint_into_model(model, state_dict, output_decoder):
    if isinstance(state_dict, dict) and "vis" in state_dict:
        model.vis.load_state_dict(strip_unused_state_dict_keys(state_dict["vis"]), strict=True)
        if model.output_decoder is not None and "flow_decoder" in state_dict:
            model.output_decoder.load_state_dict(state_dict["flow_decoder"], strict=True)
    else:
        model.vis.load_state_dict(
            strip_unused_state_dict_keys(state_dict),
            strict=output_decoder == "vae",
        )


def build_checkpoint_state(model_state_dict, output_decoder, has_output_decoder):
    vis_state_dict, decoder_state_dict = split_wrapped_state_dict(model_state_dict)
    if not has_output_decoder:
        return vis_state_dict
    return {
        "vis": vis_state_dict,
        "flow_decoder": decoder_state_dict,
        "output_decoder": output_decoder,
    }


def get_batch_string(batch, key, index, default=""):
    values = batch.get(key)
    if values is None:
        return default
    return values[index]


def should_skip_mask_gt(batch, index):
    dataset_name = get_batch_string(batch, "dataset_name", index)
    benchmark_name = get_batch_string(batch, "benchmark_name", index, default=dataset_name)
    return dataset_name in MASK_GT_SKIP_DATASETS or benchmark_name in MASK_GT_SKIP_BENCHMARKS


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # ------------------------------------------------------------------
    # 多机: 在 Accelerator() 之前手动初始化 torch.distributed(【不传 device_id】)。
    # 规避 DeepSpeed 0.19.2 在 torch 2.9.1 下给 init_process_group 传 device_id ->
    # NCCL eager_connect 全 mesh 连接在 RoCE 多机死锁。先用 torch 原生 init(惰性建连),
    # DeepSpeed/accelerate 检测到已初始化会 short-circuit 跳过带 device_id 的 init。
    # 单机(无 RANK 环境)不触发, 保持原行为。详见 train_flow_flux_roma.py 同款修复。
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        if int(os.environ["RANK"]) == 0:
            print(f">>> [dist] pre-init torch.distributed (no device_id) "
                  f"world={os.environ['WORLD_SIZE']} —— 规避 device_id eager_connect 死锁", flush=True)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulation_steps,
        log_with='wandb' if not args.debug and args.report_to == "wandb" else None,
    )
    if accelerator.is_main_process and not args.debug:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Modules of T2I diffusion models
    vae = AutoencoderKLFlux2.from_pretrained(args.diffusion_model, subfolder="vae", torch_dtype=weight_dtype)
    vis = Flux2Transformer2DModel.from_pretrained(args.diffusion_model, subfolder="transformer", torch_dtype=weight_dtype)
    flow_decoder = build_output_decoder(args.output_decoder)
    if args.output_decoder == "tiny_sd3" and args.taesd_decoder_ckpt:
        flow_decoder.load_taesd_weights(args.taesd_decoder_ckpt)
    model = FluxFlowModel(vis, flow_decoder)
    # vis = Transformer2DModel.from_config(vis.config)
    # print('random init ...')
        
    # if args.input_mode == 'single':
    #     if accelerator.is_main_process:
    #         print(">>> [Model] Configuring input layer for 'single' mode (A,B,Flow).")
    #     new_proj = torch.nn.Conv2d(12, vis.inner_dim, kernel_size=(2, 2), stride=(2, 2))
    # else: # 'double' mode
    #     if accelerator.is_main_process:
    #         print(">>> [Model] Configuring input layer for 'double' mode (Flow | A,B).")
    #     new_proj = torch.nn.Conv2d(8, vis.inner_dim, kernel_size=(2, 2), stride=(2, 2))
    # new_proj.weight.data[:, -4:] = vis.pos_embed.proj.weight.data # 预训练权重给 noisy 部分
    # new_proj.weight.data[:, :-4] = 0 # 条件部分初始化为0
    # new_proj.bias.data = vis.pos_embed.proj.bias.data
    # del vis.pos_embed.proj
    # vis.pos_embed.proj = new_proj.to(weight_dtype)

    if args.repre_model != "":
        repre_model = AutoModel.from_pretrained(args.repre_model, torch_dtype=weight_dtype)
        repre_processor = AutoImageProcessor.from_pretrained(args.repre_model)
        model.vis.dino_proj = torch.nn.Linear(1024, model.vis.inner_dim, dtype=weight_dtype)
        model.vis.dino_proj.weight.data.zero_()
        model.vis.dino_proj.bias.data.zero_()
    else:
        repre_model = None

    if args.init_ckpt_path:
        print(f">>> [Model] Loading pre-trained weights from {args.init_ckpt_path} ...")
        init_state = load_checkpoint(args.init_ckpt_path)
        load_checkpoint_into_model(model, init_state, args.output_decoder)
        if accelerator.is_main_process:
            print(f">>> [Model] Loaded initialization checkpoint with output decoder '{args.output_decoder}'.")
    ema_model = copy.deepcopy(model) if args.use_ema else None

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.diffusion_model, subfolder="tokenizer", torch_dtype=weight_dtype)
    text_encoder = Qwen3ForCausalLM.from_pretrained(args.diffusion_model, subfolder="text_encoder", torch_dtype=weight_dtype)

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.diffusion_model, subfolder="scheduler", torch_dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    if flow_decoder is not None:
        flow_decoder.requires_grad_(True)
    if ema_model is not None:
        ema_model.requires_grad_(False)
        ema_model.eval()
    if repre_model is not None:
        repre_model.requires_grad_(False)

    if args.gradient_checkpointing:
        model.vis.enable_gradient_checkpointing()

    flux_pipeline = Flux2KleinPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        transformer=None, scheduler=noise_scheduler,
    )

    # =================================================================================
    # --- 1. Dataloader 集成 (与之前版本保持一致) ---
    # =================================================================================
    if accelerator.is_main_process:
        print(">>> [Dataloader] Building unified dataloader...")

    IMG_HEIGHT, IMG_WIDTH = args.img_height, args.img_width
    if IMG_HEIGHT % 16 != 0 or IMG_WIDTH % 16 != 0:
        raise ValueError(
            f"--img_height/--img_width must be divisible by 16 for FLUX latent packing, "
            f"got {IMG_HEIGHT}x{IMG_WIDTH}."
        )
    num_workers = args.num_workers
    dataset_configs = load_dataset_config(args.dataset_config)
    if args.recipe_common_clean:
        if args.dataset_sources:
            requested_sources = {item.strip() for item in args.dataset_sources.split(",") if item.strip()}
            unknown_sources = requested_sources - set(dataset_configs)
            if unknown_sources:
                raise ValueError(f"Unknown --dataset_sources: {sorted(unknown_sources)}")
            dataset_configs = {name: cfg for name, cfg in dataset_configs.items() if name in requested_sources}
        if args.max_train_steps_override is not None and args.num_samples_per_epoch is None:
            args.num_samples_per_epoch = (
                args.max_train_steps_override
                * args.batch_size
                * accelerator.num_processes
                * accelerator.gradient_accumulation_steps
            )

    dataloader_builder = UnifiedDataloaderBuilder(
        batch_size=args.batch_size,
        num_workers=num_workers,
        img_size=(IMG_HEIGHT, IMG_WIDTH),
        num_samples_per_epoch=args.num_samples_per_epoch,
        dataset_configs=dataset_configs,
        seed=args.seed if args.recipe_common_clean else 0,
    )

    train_dataloader, train_sampler = dataloader_builder.build()
    
    num_samples_per_epoch = train_dataloader.sampler.num_samples
    args.max_train_steps = num_samples_per_epoch * args.max_train_epochs // \
                           (args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps)
    if args.max_train_steps_override is not None:
        args.max_train_steps = int(args.max_train_steps_override)

    if accelerator.is_main_process:
        print(f">>> [Dataloader] Dataloader built. Each epoch will draw {num_samples_per_epoch} samples.")
        print(f">>> [Training] Total estimated training steps: {args.max_train_steps}")
    
    test_dataloader = None 
    
    # =================================================================================
    
    optimizer_param_groups = [{"params": list(model.vis.parameters()), "lr": args.learning_rate}]
    if flow_decoder is not None:
        optimizer_param_groups.append({"params": list(model.output_decoder.parameters()), "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(optimizer_param_groups, lr=args.learning_rate, betas=(0.9, 0.999),
                                  weight_decay=1e-2, eps=1e-8)
    lr_scheduler = get_scheduler("constant_with_warmup", optimizer=optimizer,
                                 num_warmup_steps=args.warmup_steps * accelerator.num_processes,
                                 num_training_steps=args.max_train_steps * accelerator.num_processes)

    vae.to(accelerator.device, weight_dtype)
    text_encoder.to(accelerator.device, weight_dtype)
    vae.eval()
    text_encoder.eval()
    if repre_model is not None:
        repre_model.to(accelerator.device, weight_dtype)
        repre_model.eval()
    model.to(accelerator.device, weight_dtype)
    if ema_model is not None:
        ema_model.to(accelerator.device, weight_dtype)

    with torch.no_grad():
        encoder_hidden_states, text_ids = flux_pipeline.encode_prompt([""], num_images_per_prompt=args.batch_size,
                                                                      max_sequence_length=args.token_length or tokenizer.model_max_length)
        # tokens = tokenizer(["", ] * args.batch_size, max_length=args.token_length or tokenizer.model_max_length,
        #                         padding="max_length", truncation=True, return_tensors="pt").to(accelerator.device)
        # encoder_hidden_states = text_encoder(tokens.input_ids)[0].detach()
    del text_encoder
    del flux_pipeline.text_encoder
    torch.cuda.empty_cache()
    
    # Pass the dataloader so DeepSpeed can infer train_micro_batch_size_per_gpu,
    # but deliberately discard Accelerate's prepared/sharded loader. The loop
    # below keeps using the original loader, whose sampler is already distributed.
    model, optimizer, _, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler,
    )

    # # 无条件 latents
    # if args.input_mode == 'single':
    #     uncond_image_tensor = torch.zeros([1, 3, IMG_HEIGHT * 3, IMG_WIDTH], device=accelerator.device, dtype=weight_dtype)
    # else:  # double
    #     uncond_image_tensor = torch.zeros([1, 3, IMG_HEIGHT * 2, IMG_WIDTH], device=accelerator.device, dtype=weight_dtype)
    # # uncond_image_latents = vae.encode(uncond_image_tensor).latent_dist.sample()
    # uncond_image_latents = vae.encode(uncond_image_tensor).latent_dist.mode()


    global_step = 0
    start_epoch = 0
    resume_loss_l1_mae = 0.05
    wandb_id = wandb.util.generate_id()

    if args.resume and not args.debug:
        state_file = os.path.join(args.output_dir, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            
            global_step = state["global_step"]
            resume_loss_l1_mae = float(state.get("loss_l1_mae", 0.05))
            wandb_id = state["wandb_id"]
            model_checkpoint_path = os.path.join(args.output_dir, f"model_step_{global_step}.pt")

            if os.path.exists(model_checkpoint_path):
                if accelerator.is_main_process:
                    print(f">>> [Resume] Loading model from: {model_checkpoint_path}")
                
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = load_checkpoint(model_checkpoint_path)
                load_checkpoint_into_model(unwrapped_model, state_dict, args.output_decoder)
                if ema_model is not None:
                    ema_checkpoint_path = os.path.join(args.output_dir, f"ema_model_step_{global_step}.pt")
                    if os.path.exists(ema_checkpoint_path):
                        if accelerator.is_main_process:
                            print(f">>> [Resume] Loading EMA model from: {ema_checkpoint_path}")
                        ema_state_dict = load_checkpoint(ema_checkpoint_path)
                        load_checkpoint_into_model(ema_model, ema_state_dict, args.output_decoder)
                    else:
                        ema_model.load_state_dict(unwrapped_model.state_dict(), strict=True)
                
                # 快进学习率调度器
                if accelerator.is_main_process:
                    print(f">>> [Resume] Fast-forwarding LR scheduler to step {global_step}")
                lr_scheduler_last_epoch = state.get("lr_scheduler_last_epoch", 0)
                lr_scheduler.scheduler.last_epoch = lr_scheduler_last_epoch
                # for _ in range(global_step * accelerator.num_processes):
                #     lr_scheduler.step()

                # ★★★ 简化点：只计算开始的 epoch，不再计算要跳过的 batch ★★★
                steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
                start_epoch = global_step // steps_per_epoch

                if accelerator.is_main_process:
                    print(f">>> [Resume] Resuming from global_step: {global_step}, starting at epoch: {start_epoch}")
            else:
                if accelerator.is_main_process:
                    print(f"!!! [Warning] State file found but model checkpoint not found. Starting from scratch.")
                global_step = 0 # 重置
        else:
            if accelerator.is_main_process:
                print(">>> [Resume] --resume flag was passed, but 'training_state.json' not found. Starting from scratch.")

    vis_trainable = set_vis_trainable_for_decoder_warmup(
        accelerator.unwrap_model(model),
        enabled=flow_decoder is None or global_step >= args.decoder_warmup_steps,
    )
    if accelerator.is_main_process and flow_decoder is not None:
        phase = "joint DiT+decoder" if vis_trainable else "decoder warmup"
        print(f">>> [Training] Initial phase: {phase} (global_step={global_step}, warmup={args.decoder_warmup_steps})")

    if accelerator.is_main_process:
        if not args.debug and args.report_to == "wandb":
            tracker_config = dict(vars(args))
            accelerator.init_trackers(project_name='freematching', config=tracker_config,
                                      init_kwargs={'wandb': {'name': args.output_dir.split('/')[-1],
                                                             'resume': 'allow', 'id': wandb_id}, })

            # # # adapter.eval()
            # vis.eval()
            # sample_images(test_latents, vae, 
            #             [encoder_hidden_states_t5, tokens_t5.attention_mask.detach(), uncond_img_tensor], 
            #             vis, noise_scheduler, args, accelerator, weight_dtype,
            #             sample_step=20, global_step=0)

        print(f"Num examples = {train_dataloader.sampler.num_samples}")
        print(f"Total batch size = {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
        # print(f"Num Epochs = {num_train_epochs}")
        print(f"Total optimization steps = {args.max_train_steps}")
        print(f"Loss type = {args.loss_type}")

    # match_score(uncond_image_tensor, None)

    accelerator.wait_for_everyone()
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True)
    # ... (进度条和恢复逻辑) ...

    # =================================================================================
    # --- 2. 训练循环 ---
    # =================================================================================
    loss_l1_mae = resume_loss_l1_mae
    for epoch in range(start_epoch, args.max_train_epochs):
        
        if hasattr(train_dataloader.sampler, "set_epoch"):
             train_dataloader.sampler.set_epoch(epoch)

        grid = None
        model.train()

        for _, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                should_train_vis = flow_decoder is None or global_step >= args.decoder_warmup_steps
                if should_train_vis != vis_trainable:
                    vis_trainable = set_vis_trainable_for_decoder_warmup(
                        accelerator.unwrap_model(model),
                        enabled=should_train_vis,
                    )
                    if accelerator.is_main_process:
                        print(f">>> [Training] Switching to joint DiT+decoder training at step {global_step}.")

                # print(lr_scheduler.scheduler.last_epoch)
                with torch.no_grad():
                    # print(args.local_rank, batch['index'])
                    # accelerator.wait_for_everyone()
                    # continue
                    
                    # --- 从 batch 准备基础张量 ---
                    swap_flag = random.random() < 0.5
                    swap_flag = False
                    im_A = batch['im_A'].to(accelerator.device, weight_dtype)
                    im_B = batch['im_B'].to(accelerator.device, weight_dtype)
                    if swap_flag:
                        im_A, im_B = im_B, im_A

                    gt_flow_pixels = batch['ground_truth_flow'].to(accelerator.device, weight_dtype)
                    gt_mask = batch['validity_mask'].to(accelerator.device, weight_dtype)

                    B, _, H, W = gt_flow_pixels.shape
                    device = gt_flow_pixels.device

                    if grid is None:
                        grid_y, grid_x = torch.meshgrid(
                            torch.arange(H, device=device), 
                            torch.arange(W, device=device), 
                            indexing='ij'
                        )
                        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).float()

                    corr_map_pixels = gt_flow_pixels + grid

                    # 4. 将 [0, W-1] 和 [0, H-1] 范围的像素坐标归一化到 [-1, 1]
                    norm_tensor = torch.tensor([W - 1, H - 1], device=device).view(1, 2, 1, 1)
                    corr_map_normalized = (corr_map_pixels / norm_tensor).clamp(0, 1)
                    gt_mask = gt_mask * (1 - torch.isnan(corr_map_normalized).any(1).float())
                    corr_map_normalized = torch.nan_to_num(corr_map_normalized, nan=0.0)

                    flow_mask_ref = gt_mask.clone()
                    mask_mask_ref = torch.ones_like(gt_mask)
                    for i in range(gt_flow_pixels.shape[0]):
                        # if batch['dataset_name'][i] in ['megadepth', 'kubric', 'uno', 'uno_ms']:
                        #         corr_map_normalized[i] = flow_refine[i]
                        #         gt_mask[i] = mask_refine[i]
                        if should_skip_mask_gt(batch, i):
                            mask_mask_ref[i] = torch.zeros_like(gt_mask[i])

                        if batch['dataset_name'][i] == 'blender':
                            flow_mask_ref[i] = expand_mask_boundary(gt_mask[i][None, None], kernel_size=23)[0, 0].bool()

                    if repre_model is not None:
                        im_A_repre = repre_processor(
                            im_A,
                            return_tensors="pt",
                            do_resize=True,
                            size={"height": H, "width": W},
                            do_rescale=False,
                            resample=Image.Resampling.BILINEAR,
                        ).pixel_values.to(accelerator.device, weight_dtype)
                        im_B_repre = repre_processor(
                            im_B,
                            return_tensors="pt",
                            do_resize=True,
                            size={"height": H, "width": W},
                            do_rescale=False,
                            resample=Image.Resampling.BILINEAR,
                        ).pixel_values.to(accelerator.device, weight_dtype)
                        im_A_repre = repre_model(pixel_values=im_A_repre).last_hidden_state[:, 5:, :]
                        im_B_repre = repre_model(pixel_values=im_B_repre).last_hidden_state[:, 5:, :]
                    else:
                        im_A_repre = im_B_repre = None

                    # debug_warping(im_A.float(), im_B.float(), corr_map_normalized.float() * 2 - 1, gt_mask.bool())
                    # continue
                    flow_gt_tensor = torch.cat([corr_map_normalized, gt_mask[:, None]], dim=1).to(weight_dtype)
                    # mask_gt_tensor = torch.cat([torch.zeros_like(corr_map_normalized), gt_mask[:, None]], dim=1).to(weight_dtype)
                    
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (im_A.shape[0],),
                                              device=im_A.device).long()
                    if args.sample_mode == 'one':
                        timesteps = torch.ones_like(timesteps) * args.timestep
                    
                    # --- ★★★ 根据 input_mode 准备 VAE 输入和 Latents ★★★ ---
                    input_tensor = torch.cat([im_A, im_B, ], dim=0) * 2 - 1
                    with torch.no_grad():
                        input_latent, input_image_ids = flux_pipeline.prepare_image_latents_parallel(
                            input_tensor,
                            2,
                            input_tensor.device,
                            input_tensor.dtype,
                        )
                    c_ = input_latent.shape[-1]
                    a_latent, b_latent, = input_latent.chunk(2, dim=1)
                    a_latent = a_latent.view(B, -1, c_)
                    b_latent = b_latent.view(B, -1, c_)
                    input_latent = torch.cat([a_latent, b_latent, ], dim=1)

                    # latents = vae.encode(input_tensor).latent_dist.sample()
                    # latents = vae.encode(input_tensor).latent_dist.mode()
                    # latents = latents * vae.config.scaling_factor

                    # # --- 无条件引导 (Classifier-Free Guidance) ---
                    # uncond_index = random.randint(0, 9)
                    # if uncond_index < args.batch_size:
                    #     # 无论哪种模式, 都将条件（source/target latents）置为零
                    #     uncond_s, uncond_t = uncond_image_latents.chunk(2, dim=2)
                    #     source_latent[uncond_index] = uncond_s[0]
                    #     target_latent[uncond_index] = uncond_t[0]

                    # --- 加噪过程 ---
                    # noise = torch.randn_like(flow_latent)
                    # # noise = torch.zeros_like(flow_latent)

                    # # --- ★★★ 根据 input_mode 准备模型输入 ★★★ ---
                    # if args.input_mode == 'single':
                    #     noisy_model_input = torch.cat([a_latent, b_latent, noise], dim=1)
                    # else:  # 'double'
                    #     a_latent_ = torch.cat([a_latent, torch.zeros_like(noise), ], dim=1)
                    #     b_latent_ = torch.cat([b_latent, noise, ], dim=1)
                    #     noisy_model_input = torch.cat([a_latent_, b_latent_], dim=2)
                        
                # --- 前向传播 ---
                pred_out = model(
                    hidden_states=input_latent,
                    timestep=timesteps / 1000,
                    encoder_hidden_states=encoder_hidden_states[:B],
                    txt_ids=text_ids[:B],
                    img_ids=input_image_ids,
                    dino_repre=[im_A_repre, im_B_repre],
                    flux_pipeline=flux_pipeline,
                    h_flux=H // 16,
                    w_flux=W // 16,
                    output_index=0 if swap_flag else 1,
                    train_vis=vis_trainable,
                )

                # --- 损失计算 ---
                # if args.loss_type == "eps":
                #     target = noise
                # elif args.loss_type == "vlc":
                #     target = noise_scheduler.get_velocity(flow_latent, noise, timesteps)
                # else: # "x0"
                #     target = flow_latent
                
                assert args.loss_type == 'x0'

                pred_flow = pred_out[:, :2, :, :]
                pred_mask = pred_out[:, 2:3, :, :]

                # print('visual debug ...')
                # plt.imsave('vis_out/tmp_A.webp', (rearrange(im_A, 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # plt.imsave('vis_out/tmp_B.webp', (rearrange(im_B, 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # plt.imsave('vis_out/tmp_mask.png', (rearrange((pred_mask*0.5+0.5).expand(-1, 3, -1, -1), 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # warped_image = F.grid_sample(
                #         im_A, pred_flow.clamp(-1, 1).permute(0, 2, 3, 1),  # 形状已经是 (B, H, W, 2)
                #         mode='bilinear', padding_mode='border',  # 边界像素填充模式
                #         align_corners=True      # 坐标和像素角点对齐
                #     )
                # warped_image_ref = F.grid_sample(
                #         im_A, (corr_map_normalized.to(im_A)*2-1).clamp(-1, 1).permute(0, 2, 3, 1),  # 形状已经是 (B, H, W, 2)
                #         mode='bilinear', padding_mode='border',  # 边界像素填充模式
                #         align_corners=True      # 坐标和像素角点对齐
                #     )
                # plt.imsave(f'vis_out/tmp_warped.webp', (rearrange(warped_image, 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # plt.imsave(f'vis_out/tmp_warped_ref.webp', (rearrange(warped_image_ref, 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # plt.imsave(f'vis_out/tmp_flow.webp', (rearrange(torch.cat([pred_flow*0.5+0.5, torch.zeros_like(pred_mask)],dim=1), 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # plt.imsave(f'vis_out/tmp_flow_ref.webp', (rearrange(torch.cat([corr_map_normalized.to(im_A), torch.zeros_like(pred_mask)],dim=1), 'b c h w -> (b h) w c')).float().clamp(0,1).detach().cpu().numpy())  
                # print('source:', batch['dataset_name'])
                # continue

                # loss = F.mse_loss(pred_flow.float(), (corr_map_normalized*2-1).float(), reduction="none") * flow_mask_ref[:, None]
                # loss = (loss.sum(dim=(1, 2, 3)) / (flow_mask_ref.sum(dim=(1, 2)) + 1e-6)).mean()
                with torch.no_grad():
                    loss_l1 = (F.l1_loss(pred_flow.float(), (corr_map_normalized*2-1).float(), reduction="none") * flow_mask_ref[:, None]).detach()
                    loss_l1 = (loss_l1.sum(dim=(1, 2, 3)) / (flow_mask_ref.sum(dim=(1, 2)) + 1e-6)).mean()
                    gathered_loss_l1 = accelerator.gather_for_metrics(loss_l1).mean().item()
                    loss_l1_mae = 0.998 * loss_l1_mae + 0.002 * gathered_loss_l1

                loss_h = F.huber_loss(pred_flow.float(), (corr_map_normalized*2-1).float(), reduction="none", delta=loss_l1_mae * 2) * flow_mask_ref[:, None]
                loss_h = (loss_h.sum(dim=(1, 2, 3)) / (flow_mask_ref.sum(dim=(1, 2)) + 1e-6)).mean()

                loss_grad = flow_gradient_loss(pred_flow.float(), (corr_map_normalized*2-1).float(), mask=flow_mask_ref[:, None])
                loss_mask = (F.mse_loss(pred_mask.float(), (gt_mask[:, None]*2-1).float(), reduction="none") * mask_mask_ref[:, None]).mean()
                # debug_list.append((loss.sum(dim=(1, 2, 3)) / (flow_mask_ref.sum(dim=(1, 2)) + 1e-6)).mean().item())
                # gathered_loss = accelerator.gather_for_metrics(loss)
                # gathered_loss_mask = accelerator.gather_for_metrics(loss_mask)
                loss = loss_h * 10 + loss_mask * 0.33  + loss_grad * 1.0

                # if len(debug_list) == 6:
                #     print('loss:', debug_list, sum(debug_list)/len(debug_list))
                #     print('source:', batch['dataset_name'])
                # if accelerator.is_main_process:
                # accelerator.wait_for_everyone()
                # continue

                # --- 后向传播与优化 ---
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                if args.use_ema and accelerator.sync_gradients:
                    update_ema_model(ema_model, accelerator.unwrap_model(model), args.ema_decay)

            # --- 日志与保存 ---
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % 5 == 0:
                        # accuracy = (pos_pred.argmax(dim=1) == pos_label.view(-1)).float().mean()
                        accelerator.log({"train_loss": loss_h.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                                         "loss_grad": loss_grad.detach().item(), "error": loss_l1_mae,
                                         "train_vis": float(vis_trainable)},
                                         step=global_step)

                    # if global_step % args.validation_steps == 0:
                    #     # adapter.eval()
                    #     vis.eval()
                    #     # sample_images(vae, adapter, caption2embed_simple, vis, noise_scheduler,
                    #     #                   args, accelerator, weight_dtype, global_step)
                    #     sample_images(test_latents, vae, [encoder_hidden_states_t5, tokens_t5.attention_mask.detach(), uncond_img_tensor], 
                    #                   vis, noise_scheduler, args, accelerator, weight_dtype, 25, global_step)
                    #     # adapter.train()
                    #     vis.train()

                if global_step % args.save_steps == 0 or global_step >= args.max_train_steps:
                    unwrapped_model = accelerator.unwrap_model(model)
                    state_dict = build_checkpoint_state(
                        accelerator.get_state_dict(model),
                        args.output_decoder,
                        unwrapped_model.output_decoder is not None,
                    )
                    if ema_model is not None:
                        ema_state_dict = build_checkpoint_state(
                            ema_model.state_dict(),
                            args.output_decoder,
                            ema_model.output_decoder is not None,
                        )
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        model_save_path = os.path.join(args.output_dir, f"model_step_{global_step}.pt")
                        torch.save(state_dict, model_save_path)
                        print(f">>> [Save Model] Saved model weights to {model_save_path}")
                        ema_model_save_path = None
                        if ema_model is not None:
                            ema_model_save_path = os.path.join(args.output_dir, f"ema_model_step_{global_step}.pt")
                            torch.save(ema_state_dict, ema_model_save_path)
                            print(f">>> [Save Model] Saved EMA model weights to {ema_model_save_path}")

                        state = {
                            "global_step": global_step, "wandb_id": wandb_id,
                            "lr_scheduler_last_epoch": lr_scheduler.scheduler.last_epoch,
                            "model_checkpoint": os.path.basename(model_save_path),
                        }
                        if ema_model_save_path is not None:
                            state["ema_model_checkpoint"] = os.path.basename(ema_model_save_path)
                        state_file = os.path.join(args.output_dir, "training_state.json")
                        with open(state_file, "w") as f:
                            json.dump(state, f, indent=4)
                        print(f">>> [Save State] Saved training state to {state_file}")

                        model_ckpt_files = [f for f in os.listdir(args.output_dir) if f.startswith("model_step_") and f.endswith(".pt")]
                        model_ckpt_files = sorted(model_ckpt_files, key=lambda x: int(re.search(r'\d+', x).group()))
                        
                        # if len(model_ckpt_files) > args.max_saved_models:
                        #     for old_ckpt in model_ckpt_files[:-args.max_saved_models]:
                        #         os.remove(os.path.join(args.output_dir, old_ckpt))
                        #         print(f">>> [Cleanup] Removed old model checkpoint: {old_ckpt}")

                if global_step >= args.max_train_steps:
                    break
        
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
