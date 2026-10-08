"""
FLUX training with EMA teacher + RoMa weak refinement.
"""

import os
import re
import math
import copy
import json
import random
import argparse
from datetime import timedelta

import wandb.util
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from freematching.vendor.model_flux2 import Flux2Transformer2DModel
from freematching.vendor.pipeline_flux2_klein import Flux2KleinPipeline
from freematching.checkpoint import load_checkpoint
from freematching.datasets.unified_loader import load_dataset_config
from transformers import AutoImageProcessor, AutoModel, Qwen2TokenizerFast, Qwen3ForCausalLM

from freematching.datasets.unified_loader import UnifiedDataloaderBuilder, DatasetWrapper, collate_skip_none
from freematching.datasets.uno import UNOBuilder
from freematching.datasets.omniGen2 import DEFAULT_OMNIGEN2_ROOT, OmniGen2VideoWeakBuilder
from freematching.datasets.spring import SpringBuilder
from freematching.datasets.flyingthings import FlyingThingsBuilder
from freematching.training.pair_datasets import FlowWeakPairDataset, FrozenPairDataset
from freematching.training.sampling import build_source_sample_weights, parse_source_ratios
from freematching.training.losses import flow_gradient_loss
from freematching.training.utils import expand_mask_boundary
from freematching.flow_decoder import FluxFlowModel, build_output_decoder
from freematching.training.teacher import compute_iterative_roma_targets, get_roma_estimator


UNO_LABEL_ROOT = 'data/datasets/UNO-1M/labels'
DEFAULT_INIT_CKPT = "checkpoints/stage1/model.safetensors"
MASK_GT_SKIP_DATASETS = {"tracethings", "pointodyssey", "dynamic_replica", "flyingchairs"}
MASK_GT_SKIP_BENCHMARKS = {"kitti", "eth3d_dense"}


def get_batch_string(batch, key, index, default=""):
    values = batch.get(key)
    if values is None:
        return default
    return values[index]


def should_skip_mask_gt(batch, index):
    dataset_name = get_batch_string(batch, "dataset_name", index)
    benchmark_name = get_batch_string(batch, "benchmark_name", index, default=dataset_name)
    return dataset_name in MASK_GT_SKIP_DATASETS or benchmark_name in MASK_GT_SKIP_BENCHMARKS


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX training with RoMa weak refinement")
    parser.add_argument("--diffusion_model", type=str, default="black-forest-labs/FLUX.2-klein-base-4B")
    parser.add_argument("--repre_model", type=str, default="facebook/dinov3-vitl16-pretrain-lvd1689m")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--ckpt_dir", type=str, default="")
    parser.add_argument("--init_ckpt_path", type=str, default=DEFAULT_INIT_CKPT)
    parser.add_argument("--input_mode", type=str, default="double", choices=["double"])
    parser.add_argument("--sample_mode", type=str, default="one", choices=["one"])
    parser.add_argument("--use_roma_weak_sup", action="store_true")
    parser.add_argument(
        "--weak_data_sources",
        type=str,
        default="uno",
        help="Comma-separated weak-supervision sources: uno,omnigen2,flow,matching.",
    )
    parser.add_argument("--uno_label_root", type=str, default=UNO_LABEL_ROOT)
    parser.add_argument("--uno_manifest", type=str, default="", help="Optional frozen UNO pair manifest.")
    parser.add_argument("--uno_image_root", type=str, default="")
    parser.add_argument("--omnigen2_root", type=str, default=DEFAULT_OMNIGEN2_ROOT)
    parser.add_argument("--flow_weak_manifest", type=str, default="")
    parser.add_argument("--flow_image_root", type=str, default="")
    parser.add_argument("--matching_weak_manifest", type=str, default="")
    parser.add_argument("--matching_image_root", type=str, default="")
    parser.add_argument(
        "--flow_test_mix",
        action="store_true",
        help="Use image-only Spring-test + FlyingThings-test with natural per-pair mixing.",
    )
    parser.add_argument(
        "--spring_flow_root",
        type=str,
        default='data/datasets/spring/spring_processed_ufm',
    )
    parser.add_argument(
        "--flyingthings_flow_root",
        type=str,
        default='data/datasets/flyingthings/flyingthings_preprocessed_ufm',
    )
    parser.add_argument("--weak_sampling", choices=["legacy", "weighted"], default="legacy")
    parser.add_argument("--weak_data_ratios", type=str, default="", help="Example: uno:0.5,flow:0.5")
    parser.add_argument("--weak_num_samples", type=int, default=None)
    parser.add_argument("--weak_seed", type=int, default=20260724)
    parser.add_argument(
        "--weak_only",
        action="store_true",
        help="Run only weak adaptation steps. Default is off, preserving legacy Stage-2.",
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples_per_epoch", type=int, default=None)
    parser.add_argument("--max_train_epochs", type=int, default=100)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--sup_steps_per_cycle", type=int, default=3)
    parser.add_argument("--weak_steps_per_cycle", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--roma_refine_alpha", type=float, default=1.0)
    parser.add_argument("--sup_roma_loss_weight", type=float, default=0.5)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--freeze_vis", action="store_true", help="Freeze the FLUX transformer and train only the output decoder.")
    parser.add_argument("--freeze_output_decoder", action="store_true", help="Freeze the output decoder/newvae and train the FLUX transformer.")

    parser.add_argument("--token_length", type=int, default=2)
    # parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--img_height", type=int, default=288, help="Training image height. Must be divisible by 16 for FLUX latent packing.")
    parser.add_argument("--img_width", type=int, default=512, help="Training image width. Must be divisible by 16 for FLUX latent packing.")
    parser.add_argument("--loss_type", type=str, default="x0", choices=["x0"])
    parser.add_argument("--output_decoder", type=str, default="flow", choices=["vae", "tiny_sd3", "flow"])
    parser.add_argument("--taesd_decoder_ckpt", type=str, default="")

    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--max_saved_models", type=int, default=5)
    parser.add_argument(
        "--max_train_steps_override",
        type=int,
        default=None,
        help="Stop after this many optimizer updates.",
    )

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")

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


def repeat_batch_tensor(tensor, batch_size):
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        expand_shape = (batch_size,) + tuple(tensor.shape[1:])
        return tensor.expand(expand_shape)
    raise ValueError(f"Cannot expand tensor of shape {tuple(tensor.shape)} to batch size {batch_size}")


def strip_unused_state_dict_keys(state_dict):
    return {key: value for key, value in state_dict.items() if not key.startswith("dpt_head.")}


def split_checkpoint_state(state_dict):
    if isinstance(state_dict, dict) and "vis" in state_dict:
        return strip_unused_state_dict_keys(state_dict["vis"]), state_dict.get("flow_decoder"), state_dict.get("output_decoder", "flow")
    return strip_unused_state_dict_keys(state_dict), None, "vae"


def split_wrapped_state_dict(state_dict):
    vis_state = {}
    decoder_state = {}
    for key, value in state_dict.items():
        if key.startswith("vis."):
            vis_state[key[len("vis."):]] = value
        elif key.startswith("output_decoder."):
            decoder_state[key[len("output_decoder."):]] = value
    return vis_state, decoder_state


def load_checkpoint_into_model(model, state_dict, expected_decoder, checkpoint_path):
    vis_state, decoder_state, checkpoint_decoder = split_checkpoint_state(state_dict)
    if checkpoint_decoder != expected_decoder:
        raise ValueError(
            f"Checkpoint decoder mismatch: checkpoint uses '{checkpoint_decoder}', "
            f"but --output_decoder is '{expected_decoder}'."
        )
    model.vis.load_state_dict(vis_state, strict=True)
    if model.output_decoder is not None:
        if decoder_state is None:
            raise ValueError(f"{checkpoint_path} does not contain flow_decoder weights.")
        model.output_decoder.load_state_dict(decoder_state, strict=True)


def build_checkpoint_state(model_state_dict, output_decoder, has_output_decoder):
    vis_state_dict, decoder_state_dict = split_wrapped_state_dict(model_state_dict)
    if not has_output_decoder:
        return vis_state_dict
    return {
        "vis": vis_state_dict,
        "flow_decoder": decoder_state_dict,
        "output_decoder": output_decoder,
    }


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def preload_roma_estimator_distributed(roma_setting, device, rank, world_size, local_rank):
    estimator = get_roma_estimator(roma_setting)
    if world_size <= 1:
        estimator._ensure_loaded(device)
        return

    # Each node usually has its own local torch.hub cache. Let only the
    # node-local rank 0 populate that cache first, then other local ranks load.
    if local_rank == 0:
        estimator._ensure_loaded(device)
    barrier()
    if local_rank != 0:
        estimator._ensure_loaded(device)
    barrier()


def main(args):
    if args.input_mode != "double":
        raise ValueError("train_flow_flux_roma.py 当前只支持 --input_mode double")

    # ------------------------------------------------------------------
    # 多机: 在 Accelerator() 之前手动初始化 torch.distributed(【不传 device_id】)。
    #
    # 为什么: torch>=2.3 的 init_process_group 若传了 device_id, 会在建组阶段立即
    #   eager_connect_single_device 建立全 rank mesh 连接; 在本集群的 RoCE 多机环境下
    #   这一步死锁(主节点卡 distributed_c10d.py:2134, 从节点已跑到 prepare 的 broadcast)。
    #   DeepSpeed 0.19.2 的 comm/torch.py 只对 2.6.0<torch<2.7.1 禁 device_id, torch 2.9.1
    #   不在该区间 -> 它照传 device_id -> eager_connect 死锁。
    # 解法: 我们先用 torch 原生 init(不传 device_id, 走惰性建连), 之后 DeepSpeed/accelerate
    #   检测到 torch.distributed 已初始化会各自 short-circuit(deepspeed/comm/comm.py:
    #   "if cdb is None and torch.distributed.is_initialized(): ... return"),
    #   从而跳过带 device_id 的那次 init。单机(无 RANK 环境)不触发, 保持原行为。
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=30),
        )
        if int(os.environ["RANK"]) == 0:
            print(f">>> [dist] pre-init torch.distributed (no device_id) "
                  f"world={os.environ['WORLD_SIZE']} —— 规避 device_id eager_connect 死锁", flush=True)

    weak_data_sources = {item.strip().lower() for item in args.weak_data_sources.split(",") if item.strip()}
    valid_weak_sources = {"uno", "omnigen2", "flow", "matching"}
    unknown_weak_sources = weak_data_sources - valid_weak_sources
    if unknown_weak_sources:
        raise ValueError(f"Unknown --weak_data_sources values: {sorted(unknown_weak_sources)}")
    if args.use_roma_weak_sup and not weak_data_sources:
        raise ValueError("启用 RoMa 弱监督时，--weak_data_sources 不能为空")
    if args.weak_only:
        if not args.use_roma_weak_sup:
            raise ValueError("--weak_only requires --use_roma_weak_sup")
        if args.max_train_steps_override is None:
            raise ValueError("--weak_only requires --max_train_steps_override for an exact exposure budget")
        args.sup_steps_per_cycle = 0
    elif args.sup_steps_per_cycle < 1:
        raise ValueError("--sup_steps_per_cycle must be >= 1")
    if args.use_roma_weak_sup and args.weak_steps_per_cycle < 1:
        raise ValueError("启用 RoMa 弱监督时，--weak_steps_per_cycle 必须 >= 1")
    if not args.use_roma_weak_sup and args.weak_steps_per_cycle != 1:
        args.weak_steps_per_cycle = 1

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.accumulation_steps,
        log_with="wandb" if not args.debug and args.report_to == "wandb" else None,
    )

    if accelerator.is_main_process and not args.debug:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae = AutoencoderKLFlux2.from_pretrained(args.diffusion_model, subfolder="vae", torch_dtype=weight_dtype)
    vis = Flux2Transformer2DModel.from_pretrained(args.diffusion_model, subfolder="transformer", torch_dtype=weight_dtype)
    flow_decoder = build_output_decoder(args.output_decoder)
    if args.output_decoder == "tiny_sd3" and args.taesd_decoder_ckpt:
        flow_decoder.load_taesd_weights(args.taesd_decoder_ckpt)
    model = FluxFlowModel(vis, flow_decoder)

    if args.repre_model != "":
        repre_model = AutoModel.from_pretrained(args.repre_model, torch_dtype=weight_dtype)
        repre_processor = AutoImageProcessor.from_pretrained(args.repre_model)
        model.vis.dino_proj = torch.nn.Linear(1024, model.vis.inner_dim, dtype=weight_dtype)
        model.vis.dino_proj.weight.data.zero_()
        model.vis.dino_proj.bias.data.zero_()
    else:
        repre_model = None
        repre_processor = None

    if args.init_ckpt_path:
        print(">>> [Model] Loading pre-trained weights ...")
        ckpt_state = load_checkpoint(args.init_ckpt_path)
        vis_state, decoder_state, checkpoint_decoder = split_checkpoint_state(ckpt_state)
        if checkpoint_decoder != args.output_decoder:
            raise ValueError(
                f"Init checkpoint decoder mismatch: checkpoint uses '{checkpoint_decoder}', "
                f"but --output_decoder is '{args.output_decoder}'."
            )
        model.vis.load_state_dict(vis_state, strict=True)
        if model.output_decoder is not None:
            if decoder_state is None:
                raise ValueError(f"{args.init_ckpt_path} does not contain flow_decoder weights.")
            model.output_decoder.load_state_dict(decoder_state, strict=True)
    ema_model = copy.deepcopy(model)

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.diffusion_model, subfolder="tokenizer", torch_dtype=weight_dtype)
    text_encoder = Qwen3ForCausalLM.from_pretrained(args.diffusion_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.diffusion_model, subfolder="scheduler", torch_dtype=weight_dtype)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    model.requires_grad_(True)
    if args.freeze_vis:
        model.vis.requires_grad_(False)
    if args.freeze_output_decoder and model.output_decoder is not None:
        model.output_decoder.requires_grad_(False)
    ema_model.requires_grad_(False)
    ema_model.eval()
    if repre_model is not None:
        repre_model.requires_grad_(False)

    if args.gradient_checkpointing:
        model.vis.enable_gradient_checkpointing()

    flux_pipeline = Flux2KleinPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
        scheduler=noise_scheduler,
    )

    if accelerator.is_main_process:
        print(">>> [Dataloader] Building unified dataloader...")

    img_height, img_width = args.img_height, args.img_width
    if img_height % 16 != 0 or img_width % 16 != 0:
        raise ValueError(
            f"--img_height/--img_width must be divisible by 16 for FLUX latent packing, "
            f"got {img_height}x{img_width}."
        )
    num_workers = 8 if not args.debug else 8
    train_dataloader = None
    num_samples_per_epoch = 0
    supervised_micro_batches_per_epoch = 0
    weak_steps_per_cycle = args.weak_steps_per_cycle if args.use_roma_weak_sup else 0
    cycles_per_epoch = 0
    total_micro_batches_per_epoch = 0
    optimizer_steps_per_epoch = 0

    if not args.weak_only:
        dataloader_builder = UnifiedDataloaderBuilder(
            dataset_configs=load_dataset_config(args.dataset_config),
            batch_size=args.batch_size,
            num_workers=num_workers,
            img_size=(img_height, img_width),
            num_samples_per_epoch=args.num_samples_per_epoch,
        )
        train_dataloader, _ = dataloader_builder.build()
        num_samples_per_epoch = train_dataloader.sampler.num_samples
        supervised_micro_batches_per_epoch = len(train_dataloader)
        cycles_per_epoch = math.ceil(supervised_micro_batches_per_epoch / args.sup_steps_per_cycle)
        total_micro_batches_per_epoch = cycles_per_epoch * (args.sup_steps_per_cycle + weak_steps_per_cycle)
        optimizer_steps_per_epoch = math.ceil(total_micro_batches_per_epoch / accelerator.gradient_accumulation_steps)

    uno_dataloader = None

    if args.use_roma_weak_sup:
        if accelerator.is_main_process:
            print(f">>> [Dataloader] Building {sorted(weak_data_sources)} dataloader for RoMa weak supervision...")
        weak_wrapped = []
        weak_dataset_sources = []

        if "uno" in weak_data_sources:
            if args.uno_manifest:
                uno_datasets = [
                    FrozenPairDataset(
                        args.uno_manifest,
                        image_root=args.uno_image_root,
                        height=img_height,
                        width=img_width,
                        dataset_name="uno",
                    )
                ]
            else:
                uno_builder = UNOBuilder(data_root=args.uno_label_root)
                uno_datasets = uno_builder.build_dataset(
                    dataset_type="composite",
                    ht=img_height,
                    wt=img_width,
                    normalize=False,
                    debug=True,
                )
            for dataset in uno_datasets:
                weak_wrapped.append(DatasetWrapper(dataset, "uno"))
                weak_dataset_sources.append("uno")

        if "omnigen2" in weak_data_sources:
            omnigen_builder = OmniGen2VideoWeakBuilder(args.omnigen2_root)
            omnigen_datasets = omnigen_builder.build_dataset(ht=img_height, wt=img_width, debug=True)
            for dataset in omnigen_datasets:
                weak_wrapped.append(DatasetWrapper(dataset, "omnigen2_video"))
                weak_dataset_sources.append("omnigen2")

        if "flow" in weak_data_sources:
            if args.flow_test_mix:
                spring_datasets = SpringBuilder(args.spring_flow_root).build_dataset(
                    split="test",
                    ht=img_height,
                    wt=img_width,
                    normalize=False,
                    debug=True,
                    image_only=True,
                    temporal_gaps=(1, 2, 4, 8),
                    bidirectional=True,
                )
                flyingthings_datasets = FlyingThingsBuilder(args.flyingthings_flow_root).build_dataset(
                    split="test",
                    ht=img_height,
                    wt=img_width,
                    normalize=False,
                    debug=True,
                    image_only=True,
                    enumerate_image_variants=True,
                    bidirectional=True,
                    aug_left_right=False,
                    aug_clean_final=False,
                )
                for dataset in spring_datasets:
                    weak_wrapped.append(DatasetWrapper(dataset, "spring_test_weak"))
                    weak_dataset_sources.append("spring_flow")
                for dataset in flyingthings_datasets:
                    weak_wrapped.append(DatasetWrapper(dataset, "flyingthings_test_weak"))
                    weak_dataset_sources.append("flyingthings_flow")
                if accelerator.is_main_process:
                    print(
                        ">>> [Flow test mix] expanded image-only datasets: "
                        f"Spring={sum(len(ds) for ds in spring_datasets)}, "
                        f"FlyingThings={sum(len(ds) for ds in flyingthings_datasets)}; "
                        "base-source ratio is controlled separately",
                        flush=True,
                    )
            else:
                if not args.flow_weak_manifest:
                    raise ValueError(
                        "--flow_weak_manifest is required for flow source unless --flow_test_mix is set."
                    )
                flow_dataset = FlowWeakPairDataset(
                    args.flow_weak_manifest,
                    image_root=args.flow_image_root,
                    height=img_height,
                    width=img_width,
                )
                weak_wrapped.append(DatasetWrapper(flow_dataset, "flow_weak"))
                weak_dataset_sources.append("flow")

        if "matching" in weak_data_sources:
            if not args.matching_weak_manifest:
                raise ValueError(
                    "--matching_weak_manifest is required when weak_data_sources contains 'matching'."
                )
            matching_dataset = FlowWeakPairDataset(
                args.matching_weak_manifest,
                image_root=args.matching_image_root,
                height=img_height,
                width=img_width,
                dataset_name="matching_weak",
            )
            weak_wrapped.append(DatasetWrapper(matching_dataset, "matching_weak"))
            weak_dataset_sources.append("matching")

        if not weak_wrapped:
            raise RuntimeError("No weak-supervision datasets were built.")

        uno_dataset = ConcatDataset(weak_wrapped)
        if args.weak_sampling == "legacy":
            # Preserve the exact pre-rebuttal behavior unless explicitly requested.
            uno_dataloader = DataLoader(
                uno_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=lambda batch: torch.utils.data.default_collate([b for b in batch if b is not None]),
            )
        else:
            source_ratios = parse_source_ratios(args.weak_data_ratios, weak_dataset_sources)
            sample_weights = build_source_sample_weights(
                [len(dataset) for dataset in weak_wrapped],
                weak_dataset_sources,
                source_ratios,
            )
            generator = torch.Generator().manual_seed(args.weak_seed)
            weak_sampler = WeightedRandomSampler(
                sample_weights,
                num_samples=args.weak_num_samples or len(uno_dataset),
                replacement=True,
                generator=generator,
            )
            uno_dataloader = DataLoader(
                uno_dataset,
                batch_size=args.batch_size,
                sampler=weak_sampler,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=collate_skip_none,
            )
            if accelerator.is_main_process:
                print(f">>> [Dataloader] Controlled weak source ratios: {source_ratios}")

    if args.weak_only:
        if uno_dataloader is None:
            raise RuntimeError("Weak-only training requires a weak-supervision dataloader.")
        args.max_train_steps = args.max_train_steps_override
        total_micro_batches_per_epoch = args.max_train_steps * accelerator.gradient_accumulation_steps
        cycles_per_epoch = math.ceil(total_micro_batches_per_epoch / weak_steps_per_cycle)
        optimizer_steps_per_epoch = args.max_train_steps
    else:
        args.max_train_steps = optimizer_steps_per_epoch * args.max_train_epochs
        if args.max_train_steps_override is not None:
            if args.max_train_steps_override < 1:
                raise ValueError("--max_train_steps_override must be positive.")
            args.max_train_steps = args.max_train_steps_override

    if accelerator.is_main_process:
        print(f">>> [Dataloader] Dataloader built. Each epoch will draw {num_samples_per_epoch} samples.")
        if uno_dataloader is not None:
            print(f">>> [Dataloader] UNO weak-supervision dataset size: {len(uno_dataloader.dataset)}")
        print(f">>> [Training] Cycle ratio (sup:weak) = {args.sup_steps_per_cycle}:{weak_steps_per_cycle}")
        print(f">>> [Training] Supervised micro batches / epoch: {supervised_micro_batches_per_epoch}")
        print(f">>> [Training] Cycles / epoch: {cycles_per_epoch}")
        print(f">>> [Training] Total micro batches / epoch: {total_micro_batches_per_epoch}")
        print(f">>> [Training] Optimizer steps / epoch: {optimizer_steps_per_epoch}")
        print(f">>> [Training] Total estimated training steps: {args.max_train_steps}")

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters left after applying freeze options.")
    if accelerator.is_main_process:
        total_params = sum(param.numel() for param in model.parameters())
        trainable_param_count = sum(param.numel() for param in trainable_params)
        print(f">>> [Training] Freeze FLUX transformer: {args.freeze_vis}")
        print(f">>> [Training] Freeze output decoder: {args.freeze_output_decoder}")
        print(f">>> [Training] Trainable parameters: {trainable_param_count}/{total_params}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2, eps=1e-8)
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=3000 * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    vae.to(accelerator.device, weight_dtype)
    text_encoder.to(accelerator.device, weight_dtype)
    model.to(accelerator.device, weight_dtype)
    ema_model.to(accelerator.device, weight_dtype)
    vae.eval()
    text_encoder.eval()
    if repre_model is not None:
        repre_model.to(accelerator.device, weight_dtype)
        repre_model.eval()

    # Keep the unified supervised dataloader unwrapped because it already owns
    # distributed weighted sampling. Only the standalone UNO loader should use
    # Accelerate's wrapped loader.
    if args.weak_only:
        model, optimizer, uno_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, uno_dataloader, lr_scheduler
        )
    elif uno_dataloader is not None:
        model, optimizer, _, uno_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, uno_dataloader, lr_scheduler
        )
    else:
        model, optimizer, _, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)

    with torch.no_grad():
        encoder_hidden_states, text_ids = flux_pipeline.encode_prompt(
            [""],
            num_images_per_prompt=1,
            max_sequence_length=args.token_length or tokenizer.model_max_length,
        )
    del text_encoder
    del flux_pipeline.text_encoder

    preload_roma_estimator_distributed(
        roma_setting="fast",
        device=accelerator.device,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        local_rank=accelerator.local_process_index,
    )

    global_step = 0
    global_sup_step = 0
    global_weak_step = 0
    start_epoch = 0
    wandb_id = wandb.util.generate_id()

    if args.resume and not args.debug:
        state_file = os.path.join(args.output_dir, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
            global_step = state["global_step"]
            wandb_id = state["wandb_id"]
            model_checkpoint_path = os.path.join(args.output_dir, f"model_step_{global_step}.pt")
            if os.path.exists(model_checkpoint_path):
                if accelerator.is_main_process:
                    print(f">>> [Resume] Loading model from: {model_checkpoint_path}")
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = load_checkpoint(model_checkpoint_path)
                load_checkpoint_into_model(unwrapped_model, state_dict, args.output_decoder, model_checkpoint_path)
                ema_checkpoint_path = os.path.join(args.output_dir, f"ema_model_step_{global_step}.pt")
                if os.path.exists(ema_checkpoint_path):
                    if accelerator.is_main_process:
                        print(f">>> [Resume] Loading EMA model from: {ema_checkpoint_path}")
                    ema_state_dict = load_checkpoint(ema_checkpoint_path)
                    load_checkpoint_into_model(ema_model, ema_state_dict, args.output_decoder, ema_checkpoint_path)
                else:
                    ema_model.load_state_dict(unwrapped_model.state_dict(), strict=True)
                    if accelerator.is_main_process:
                        print(">>> [Resume] EMA checkpoint not found; initialized EMA from std model.")
                if accelerator.is_main_process:
                    print(f">>> [Resume] Fast-forwarding LR scheduler to step {global_step}")
                lr_scheduler_last_epoch = state.get("lr_scheduler_last_epoch", 0)
                lr_scheduler.scheduler.last_epoch = lr_scheduler_last_epoch
                start_epoch = global_step // optimizer_steps_per_epoch
                if accelerator.is_main_process:
                    print(f">>> [Resume] Resuming from global_step: {global_step}, starting at epoch: {start_epoch}")
            else:
                if accelerator.is_main_process:
                    print("!!! [Warning] State file found but model checkpoint not found. Starting from scratch.")
                global_step = 0
        elif accelerator.is_main_process:
            print(">>> [Resume] --resume flag was passed, but 'training_state.json' not found. Starting from scratch.")

    if accelerator.is_main_process:
        if not args.debug and args.report_to == "wandb":
            tracker_config = dict(vars(args))
            accelerator.init_trackers(
                project_name="freematching",
                config=tracker_config,
                init_kwargs={"wandb": {"name": args.output_dir.split("/")[-1], "resume": "allow", "id": wandb_id}},
            )

        print(f"Num examples = {num_samples_per_epoch}")
        print(f"Total batch size = {args.batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps}")
        print(f"Total optimization steps = {args.max_train_steps}")
        print(f"Sup/Weak cycle = {args.sup_steps_per_cycle}:{weak_steps_per_cycle}")
        print(f"Loss type = {args.loss_type}")

    accelerator.wait_for_everyone()
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process, dynamic_ncols=True)

    loss_l1_mae_sup = 0.05
    loss_l1_mae_weak = 0.05
    grid = None

    def encode_dino_repre(im_a, im_b, height, width):
        if repre_model is None:
            return None, None
        im_a_repre = repre_processor(
            im_a,
            return_tensors="pt",
            do_resize=True,
            size={"height": height, "width": width},
            do_rescale=False,
            resample=Image.Resampling.BILINEAR,
        ).pixel_values.to(accelerator.device, weight_dtype)
        im_b_repre = repre_processor(
            im_b,
            return_tensors="pt",
            do_resize=True,
            size={"height": height, "width": width},
            do_rescale=False,
            resample=Image.Resampling.BILINEAR,
        ).pixel_values.to(accelerator.device, weight_dtype)
        im_a_repre = repre_model(pixel_values=im_a_repre).last_hidden_state[:, 5:, :]
        im_b_repre = repre_model(pixel_values=im_b_repre).last_hidden_state[:, 5:, :]
        return im_a_repre, im_b_repre

    def prepare_batch_common(batch):
        nonlocal grid
        # FLUX baseline currently trains/evaluates with a fixed A->B direction.
        # Do not enable random swap unless GT / output direction is updated too.
        swap_flag = False
        im_a = batch["im_A"].to(accelerator.device, weight_dtype)
        im_b = batch["im_B"].to(accelerator.device, weight_dtype)
        if swap_flag:
            im_a, im_b = im_b, im_a

        gt_flow_pixels = batch["ground_truth_flow"].to(accelerator.device, weight_dtype)
        gt_mask = batch["validity_mask"].to(accelerator.device, weight_dtype)
        batch_size, _, height, width = gt_flow_pixels.shape
        device = gt_flow_pixels.device

        if grid is None or grid.shape[-2:] != (height, width):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).float()

        corr_map_pixels = gt_flow_pixels + grid
        norm_tensor = torch.tensor([width - 1, height - 1], device=device).view(1, 2, 1, 1)
        corr_map_normalized = (corr_map_pixels / norm_tensor).clamp(0, 1)
        gt_mask = gt_mask * (1 - torch.isnan(corr_map_normalized).any(1).float())
        corr_map_normalized = torch.nan_to_num(corr_map_normalized, nan=0.0)

        flow_mask_ref = gt_mask.clone()
        mask_mask_ref = torch.ones_like(gt_mask)
        dataset_names = batch.get("dataset_name")
        for i in range(batch_size):
            if should_skip_mask_gt(batch, i):
                mask_mask_ref[i] = torch.zeros_like(gt_mask[i])
            if dataset_names is not None and dataset_names[i] == "blender":
                flow_mask_ref[i] = expand_mask_boundary(gt_mask[i][None, None], kernel_size=23)[0, 0].bool()

        im_a_repre, im_b_repre = encode_dino_repre(im_a, im_b, height, width)
        return {
            "swap_flag": swap_flag,
            "im_a": im_a,
            "im_b": im_b,
            "gt_mask": gt_mask,
            "corr_map_normalized": corr_map_normalized,
            "flow_mask_ref": flow_mask_ref,
            "mask_mask_ref": mask_mask_ref,
            "im_a_repre": im_a_repre,
            "im_b_repre": im_b_repre,
            "batch_size": batch_size,
            "height": height,
            "width": width,
        }

    def decode_flow_and_mask(flow_model, im_a, im_b, im_a_repre, im_b_repre, batch_size, height, width, timesteps):
        with torch.no_grad():
            input_tensor = torch.cat([im_a, im_b], dim=0) * 2 - 1
            input_latent, input_image_ids = flux_pipeline.prepare_image_latents_parallel(
                input_tensor, 2, input_tensor.device, input_tensor.dtype
            )
            channels = input_latent.shape[-1]
            a_latent, b_latent = input_latent.chunk(2, dim=1)
            a_latent = a_latent.view(batch_size, -1, channels)
            b_latent = b_latent.view(batch_size, -1, channels)
            hidden_states = torch.cat([a_latent, b_latent], dim=1)

            batch_encoder_hidden_states = repeat_batch_tensor(encoder_hidden_states, batch_size)
            batch_text_ids = repeat_batch_tensor(text_ids, batch_size)
            batch_image_ids = repeat_batch_tensor(input_image_ids, batch_size)

        pred_out = flow_model(
            hidden_states=hidden_states,
            timestep=timesteps / 1000,
            encoder_hidden_states=batch_encoder_hidden_states,
            txt_ids=batch_text_ids,
            img_ids=batch_image_ids,
            dino_repre=[im_a_repre, im_b_repre] if im_a_repre is not None else None,
            flux_pipeline=flux_pipeline,
            h_flux=height // 16,
            w_flux=width // 16,
            output_index=1,
        )
        return pred_out[:, :2], pred_out[:, 2:3]

    def build_roma_weak_targets(im_a, im_b, timesteps, im_a_repre, im_b_repre, batch_size, height, width):
        with torch.no_grad():
            ema_flow, ema_mask = decode_flow_and_mask(
                ema_model, im_a, im_b, im_a_repre, im_b_repre, batch_size, height, width, timesteps
            )
            roma_targets = compute_iterative_roma_targets(
                base_flow=ema_flow.detach(),
                source_image=im_a,
                target_image=im_b,
                roma_setting="fast",
                roma_conf_thresh=0.2,
                require_improvement=False,
                improvement_margin=0.0,
                return_masks_meta=True,
            )
            ema_mask_bin = ((ema_mask[:, :1].detach() * 0.5 + 0.5) > 0.7).float()
            weak_flow = ema_flow.detach() + args.roma_refine_alpha * (roma_targets["new_flow"].to(ema_flow) - ema_flow.detach())
            weak_flow = weak_flow.clamp(-1, 1)
            weak_mask = (roma_targets["new_mask"].to(ema_flow) * ema_mask_bin).squeeze(1)
            residual_flow = roma_targets["residual_flow"].to(ema_flow)
            stats = {
                "residual_flow": residual_flow,
                "ema_mask_ratio": ema_mask_bin.to(ema_flow).mean().detach(),
                "roma_conf_mask_ratio": (roma_targets["roma_conf"].to(ema_flow) > 0.2).float().mean().detach(),
                "improvement_mask_ratio": roma_targets["improvement_mask"].to(ema_flow).mean().detach(),
            }
        return ema_flow, ema_mask, weak_flow, weak_mask, stats

    def run_batch(batch, is_roma_weak_batch=False):
        nonlocal loss_l1_mae_sup, loss_l1_mae_weak

        with accelerator.accumulate(model):
            with torch.no_grad():
                common = prepare_batch_common(batch)
                timesteps = torch.ones((common["batch_size"],), device=accelerator.device, dtype=torch.long) * args.timestep

            metrics = {}
            if is_roma_weak_batch:
                ema_flow, ema_mask, weak_flow, weak_mask, roma_stats = build_roma_weak_targets(
                    common["im_a"],
                    common["im_b"],
                    timesteps,
                    common["im_a_repre"],
                    common["im_b_repre"],
                    common["batch_size"],
                    common["height"],
                    common["width"],
                )
                pred_flow, pred_mask = decode_flow_and_mask(
                    model,
                    common["im_a"],
                    common["im_b"],
                    common["im_a_repre"],
                    common["im_b_repre"],
                    common["batch_size"],
                    common["height"],
                    common["width"],
                    timesteps,
                )
                residual_flow = roma_stats["residual_flow"]
                residual_mag_map = torch.norm(residual_flow.float(), dim=1)
                residual_mag = residual_mag_map.mean().detach()
                weak_valid_pixels = weak_mask.sum(dim=(1, 2))
                residual_mag_masked = ((residual_mag_map * weak_mask.float()).sum(dim=(1, 2)) / (weak_valid_pixels + 1e-6)).mean().detach()
                local_mask_ratio = weak_mask.mean().detach()
                has_valid_weak = (weak_valid_pixels > 0).any().float().to(pred_flow.device)
                gathered_has_valid = accelerator.gather_for_metrics(has_valid_weak)
                valid_count = gathered_has_valid.sum().item()
                gathered_mask_ratio = accelerator.gather_for_metrics(local_mask_ratio).mean().item()
                gathered_residual_mag = accelerator.gather_for_metrics(residual_mag).mean().item()
                gathered_ema_mask_ratio = accelerator.gather_for_metrics(roma_stats["ema_mask_ratio"]).mean().item()
                gathered_roma_conf_mask_ratio = accelerator.gather_for_metrics(roma_stats["roma_conf_mask_ratio"]).mean().item()
                gathered_improvement_mask_ratio = accelerator.gather_for_metrics(roma_stats["improvement_mask_ratio"]).mean().item()
                loss_mask = F.mse_loss(pred_mask.float(), ema_mask.float(), reduction="mean")
                gathered_loss_mask = accelerator.gather_for_metrics(loss_mask.detach()).mean().item()

                if valid_count <= 0:
                    loss = pred_flow.float().sum() * 0.0
                    gathered_loss_l1 = 0.0
                    gathered_loss_h = 0.0
                    global_residual_mag_masked = 0.0
                    batch_type = "roma_weak_zero"
                else:
                    with torch.no_grad():
                        loss_l1 = (F.l1_loss(pred_flow.float(), weak_flow.float(), reduction="none") * weak_mask[:, None]).detach()
                        loss_l1 = (loss_l1.sum(dim=(1, 2, 3)) / (weak_mask.sum(dim=(1, 2)) * 2 + 1e-6)).mean()
                        gathered_loss_l1_per_rank = accelerator.gather_for_metrics(loss_l1)
                        gathered_loss_l1 = ((gathered_loss_l1_per_rank * gathered_has_valid).sum().item() / valid_count)
                        loss_l1_mae_weak = 0.998 * loss_l1_mae_weak + 0.002 * gathered_loss_l1

                    loss_h = F.huber_loss(pred_flow.float(), weak_flow.float(), reduction="none", delta=loss_l1_mae_weak * 2) * weak_mask[:, None]
                    loss_h = (loss_h.sum(dim=(1, 2, 3)) / (weak_mask.sum(dim=(1, 2)) * 2 + 1e-6)).mean()
                    loss = loss_h * 10 + loss_mask * 0.2
                    gathered_loss_h_per_rank = accelerator.gather_for_metrics(loss_h.detach())
                    gathered_loss_h = ((gathered_loss_h_per_rank * gathered_has_valid).sum().item() / valid_count)
                    gathered_residual_mag_masked = accelerator.gather_for_metrics(residual_mag_masked)
                    global_residual_mag_masked = ((gathered_residual_mag_masked * gathered_has_valid).sum().item() / valid_count)
                    batch_type = "roma_weak"

                metrics.update(
                    {
                        "batch_type": batch_type,
                        "loss_h": torch.tensor(gathered_loss_h, device=accelerator.device),
                        "error": torch.tensor(gathered_loss_l1, device=accelerator.device),
                        "error_ema": torch.tensor(loss_l1_mae_weak, device=accelerator.device),
                        "loss_mask": torch.tensor(gathered_loss_mask, device=accelerator.device),
                        "ema_mask_ratio": torch.tensor(gathered_ema_mask_ratio, device=accelerator.device),
                        "roma_conf_mask_ratio": torch.tensor(gathered_roma_conf_mask_ratio, device=accelerator.device),
                        "improvement_mask_ratio": torch.tensor(gathered_improvement_mask_ratio, device=accelerator.device),
                        "roma_mask_ratio": torch.tensor(gathered_mask_ratio, device=accelerator.device),
                        "roma_residual_mag": torch.tensor(gathered_residual_mag, device=accelerator.device),
                        "roma_residual_mag_masked": torch.tensor(global_residual_mag_masked, device=accelerator.device),
                    }
                )
            else:
                pred_flow, pred_mask = decode_flow_and_mask(
                    model,
                    common["im_a"],
                    common["im_b"],
                    common["im_a_repre"],
                    common["im_b_repre"],
                    common["batch_size"],
                    common["height"],
                    common["width"],
                    timesteps,
                )
                target_flow = common["corr_map_normalized"] * 2 - 1
                with torch.no_grad():
                    loss_l1 = (F.l1_loss(pred_flow.float(), target_flow.float(), reduction="none") * common["flow_mask_ref"][:, None]).detach()
                    loss_l1 = (loss_l1.sum(dim=(1, 2, 3)) / (common["flow_mask_ref"].sum(dim=(1, 2)) * 2 + 1e-6)).mean()
                    gathered_loss_l1 = accelerator.gather_for_metrics(loss_l1).mean().item()
                    loss_l1_mae_sup = 0.998 * loss_l1_mae_sup + 0.002 * gathered_loss_l1

                loss_h = F.huber_loss(pred_flow.float(), target_flow.float(), reduction="none", delta=loss_l1_mae_sup * 2) * common["flow_mask_ref"][:, None]
                loss_h = (loss_h.sum(dim=(1, 2, 3)) / (common["flow_mask_ref"].sum(dim=(1, 2)) * 2 + 1e-6)).mean()
                loss_grad = flow_gradient_loss(pred_flow.float(), target_flow.float(), mask=common["flow_mask_ref"][:, None])
                loss_mask = (F.mse_loss(pred_mask.float(), (common["gt_mask"][:, None] * 2 - 1).float(), reduction="none") * common["mask_mask_ref"][:, None]).mean()

                if args.sup_roma_loss_weight > 0:
                    _, _, weak_flow, weak_mask, sup_roma_stats = build_roma_weak_targets(
                        common["im_a"],
                        common["im_b"],
                        timesteps,
                        common["im_a_repre"],
                        common["im_b_repre"],
                        common["batch_size"],
                        common["height"],
                        common["width"],
                    )
                    sup_roma_valid_pixels = weak_mask.sum(dim=(1, 2))
                    sup_roma_mask_ratio = weak_mask.float().mean().detach()
                    sup_roma_residual_mag_map = torch.norm(sup_roma_stats["residual_flow"].float(), dim=1)
                    if (sup_roma_valid_pixels > 0).any():
                        with torch.no_grad():
                            sup_roma_l1 = (F.l1_loss(pred_flow.float(), weak_flow.float(), reduction="none") * weak_mask[:, None]).detach()
                            sup_roma_l1 = (sup_roma_l1.sum(dim=(1, 2, 3)) / (sup_roma_valid_pixels * 2 + 1e-6)).mean()
                        sup_roma_residual_mag_masked = (
                            (sup_roma_residual_mag_map * weak_mask.float()).sum(dim=(1, 2)) / (sup_roma_valid_pixels + 1e-6)
                        ).mean().detach()
                    else:
                        sup_roma_l1 = pred_flow.float().sum() * 0.0
                        sup_roma_residual_mag_masked = pred_flow.float().sum() * 0.0
                    if (sup_roma_valid_pixels > 0).any():
                        loss_h_sup_roma = F.huber_loss(pred_flow.float(), weak_flow.float(), reduction="none", delta=loss_l1_mae_weak * 2) * weak_mask[:, None]
                        loss_h_sup_roma = (loss_h_sup_roma.sum(dim=(1, 2, 3)) / (sup_roma_valid_pixels * 2 + 1e-6)).mean()
                    else:
                        loss_h_sup_roma = pred_flow.float().sum() * 0.0
                else:
                    zero_metric = pred_flow.float().sum() * 0.0
                    sup_roma_mask_ratio = zero_metric.detach()
                    sup_roma_residual_mag_masked = zero_metric.detach()
                    sup_roma_l1 = zero_metric.detach()
                    loss_h_sup_roma = zero_metric

                loss = loss_h * 10 + loss_mask * 0.2 + loss_grad * 1.0 + args.sup_roma_loss_weight * (loss_h_sup_roma * 10)

                metrics.update(
                    {
                        "batch_type": "supervised",
                        "loss_h": loss_h.detach(),
                        "loss_grad": loss_grad.detach(),
                        "loss_mask": loss_mask.detach(),
                        "loss_h_sup_roma": loss_h_sup_roma.detach(),
                        "sup_roma_mask_ratio": sup_roma_mask_ratio,
                        "sup_roma_residual_mag_masked": sup_roma_residual_mag_masked,
                        "sup_roma_l1": sup_roma_l1.detach(),
                        "error": torch.tensor(loss_l1_mae_sup, device=accelerator.device),
                    }
                )

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                update_ema_model(ema_model, accelerator.unwrap_model(model), args.ema_decay)

        metrics["loss"] = loss.detach()
        metrics["sync_gradients"] = accelerator.sync_gradients
        return metrics

    for epoch in range(start_epoch, args.max_train_epochs):
        if train_dataloader is not None and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        if uno_dataloader is not None and hasattr(uno_dataloader.sampler, "set_epoch"):
            uno_dataloader.sampler.set_epoch(epoch)

        model.train()
        if args.freeze_vis:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.vis.eval()
            if unwrapped_model.output_decoder is not None:
                unwrapped_model.output_decoder.train()
        if args.freeze_output_decoder:
            unwrapped_model = accelerator.unwrap_model(model)
            if unwrapped_model.output_decoder is not None:
                unwrapped_model.output_decoder.eval()
        train_iterator = iter(train_dataloader) if train_dataloader is not None else None
        uno_iterator = iter(uno_dataloader) if uno_dataloader is not None else None

        def handle_step_metrics(step_metrics, is_roma_weak_step=False):
            nonlocal global_step, global_sup_step, global_weak_step
            if not step_metrics["sync_gradients"]:
                return False

            progress_bar.update(1)
            global_step += 1
            if is_roma_weak_step:
                global_weak_step += 1
            else:
                global_sup_step += 1

            if accelerator.is_main_process and global_step % 5 == 0:
                if is_roma_weak_step:
                    accelerator.log(
                        {
                            "roma_flow_loss": step_metrics["loss_h"].item(),
                            "roma_mask_loss": step_metrics["loss_mask"].item(),
                            "roma_error": step_metrics["error"].item(),
                            "roma_error_ema": step_metrics["error_ema"].item(),
                            "ema_mask_ratio": step_metrics["ema_mask_ratio"].item(),
                            "roma_conf_mask_ratio": step_metrics["roma_conf_mask_ratio"].item(),
                            "improvement_mask_ratio": step_metrics["improvement_mask_ratio"].item(),
                            "roma_mask_ratio": step_metrics["roma_mask_ratio"].item(),
                            "roma_residual_mag": step_metrics["roma_residual_mag"].item(),
                            "roma_residual_mag_masked": step_metrics["roma_residual_mag_masked"].item(),
                            "global_step_total": global_step,
                            "global_sup_step": global_sup_step,
                            "global_weak_step": global_weak_step,
                            "lr": lr_scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )
                else:
                    accelerator.log(
                        {
                            "train_loss": step_metrics["loss_h"].item(),
                            "lr": lr_scheduler.get_last_lr()[0],
                            "loss_grad": step_metrics["loss_grad"].item(),
                            "loss_mask": step_metrics["loss_mask"].item(),
                            "loss_sup_roma": step_metrics["loss_h_sup_roma"].item(),
                            "sup_roma_mask_ratio": step_metrics["sup_roma_mask_ratio"].item(),
                            "sup_roma_residual_mag_masked": step_metrics["sup_roma_residual_mag_masked"].item(),
                            "sup_roma_l1": step_metrics["sup_roma_l1"].item(),
                            "error": step_metrics["error"].item(),
                            "global_step_total": global_step,
                            "global_sup_step": global_sup_step,
                            "global_weak_step": global_weak_step,
                        },
                        step=global_step,
                    )

            if global_step % args.save_steps == 0 or global_step >= args.max_train_steps:
                unwrapped_model = accelerator.unwrap_model(model)
                state_dict = build_checkpoint_state(
                    accelerator.get_state_dict(model),
                    args.output_decoder,
                    unwrapped_model.output_decoder is not None,
                )
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
                    ema_model_save_path = os.path.join(args.output_dir, f"ema_model_step_{global_step}.pt")
                    torch.save(ema_state_dict, ema_model_save_path)
                    print(f">>> [Save Model] Saved EMA model weights to {ema_model_save_path}")
                    state = {
                        "global_step": global_step,
                        "wandb_id": wandb_id,
                        "lr_scheduler_last_epoch": lr_scheduler.scheduler.last_epoch,
                        "model_checkpoint": os.path.basename(model_save_path),
                        "ema_model_checkpoint": os.path.basename(ema_model_save_path),
                    }
                    state_file = os.path.join(args.output_dir, "training_state.json")
                    with open(state_file, "w") as f:
                        json.dump(state, f, indent=4)
                    print(f">>> [Save State] Saved training state to {state_file}")
            return global_step >= args.max_train_steps

        for _ in range(cycles_per_epoch):
            for _ in range(args.sup_steps_per_cycle):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    train_iterator = iter(train_dataloader)
                    batch = next(train_iterator)
                if handle_step_metrics(run_batch(batch, is_roma_weak_batch=False), is_roma_weak_step=False):
                    break
            if global_step >= args.max_train_steps:
                break

            for _ in range(weak_steps_per_cycle):
                if uno_iterator is None:
                    break
                try:
                    uno_batch = next(uno_iterator)
                except StopIteration:
                    uno_iterator = iter(uno_dataloader)
                    uno_batch = next(uno_iterator)
                if handle_step_metrics(run_batch(uno_batch, is_roma_weak_batch=True), is_roma_weak_step=True):
                    break
            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
