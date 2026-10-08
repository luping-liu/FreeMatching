from types import SimpleNamespace
from pathlib import Path
import json
import torch
from PIL import Image
from diffusers import AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
from transformers import AutoImageProcessor, AutoModel, Qwen2TokenizerFast, Qwen3ForCausalLM
from .vendor.model_flux2 import Flux2Transformer2DModel
from .vendor.pipeline_flux2_klein import Flux2KleinPipeline
from .checkpoint import load_checkpoint
from .geometry import warp_source, correspondence_to_flow
from freematching.flow_decoder import build_output_decoder, decode_flux_output

def repeat_batch_tensor(tensor, batch_size):
    if tensor is None:
        return None
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        expand_shape = (batch_size,) + tuple(tensor.shape[1:])
        return tensor.expand(expand_shape)
    raise ValueError(f"Cannot repeat tensor with shape {tuple(tensor.shape)} to batch size {batch_size}")

def build_model_stack(args, device, weight_dtype, enable_dino_proj, output_decoder_kind):
    vae = AutoencoderKLFlux2.from_pretrained(args.diffusion_model, subfolder="vae", torch_dtype=weight_dtype)
    vis = Flux2Transformer2DModel.from_pretrained(args.diffusion_model, subfolder="transformer", torch_dtype=weight_dtype)
    flow_decoder = build_output_decoder(output_decoder_kind)

    if enable_dino_proj:
        vis.dino_proj = torch.nn.Linear(1024, vis.inner_dim, dtype=weight_dtype)
        vis.dino_proj.weight.data.zero_()
        vis.dino_proj.bias.data.zero_()
    else:
        vis.dino_proj = None

    state = load_checkpoint(args.checkpoint)
    vis.load_state_dict(state['vis'], strict=True)
    flow_decoder.load_state_dict(state['flow_decoder'], strict=True)
    del state

    tokenizer = Qwen2TokenizerFast.from_pretrained(args.diffusion_model, subfolder="tokenizer", torch_dtype=weight_dtype)
    text_encoder = Qwen3ForCausalLM.from_pretrained(args.diffusion_model, subfolder="text_encoder", torch_dtype=weight_dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.diffusion_model, subfolder="scheduler", torch_dtype=weight_dtype)

    repre_model = None
    repre_processor = None
    if args.repre_model:
        repre_model = AutoModel.from_pretrained(args.repre_model, torch_dtype=weight_dtype)
        repre_processor = AutoImageProcessor.from_pretrained(args.repre_model)
        repre_model.requires_grad_(False)
        repre_model.to(device, dtype=weight_dtype)
        repre_model.eval()

    flux_pipeline = Flux2KleinPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=None,
        scheduler=scheduler,
    )

    vae.requires_grad_(False)
    vis.requires_grad_(False)
    if flow_decoder is not None:
        flow_decoder.requires_grad_(False)
    vae.to(device, dtype=weight_dtype).eval()
    vis.to(device, dtype=weight_dtype).eval()
    if flow_decoder is not None:
        flow_decoder.to(device, dtype=weight_dtype).eval()
    text_encoder.to(device, dtype=weight_dtype).eval()

    with torch.no_grad():
        encoder_hidden_states, text_ids = flux_pipeline.encode_prompt(
            [""],
            num_images_per_prompt=1,
            max_sequence_length=args.token_length or tokenizer.model_max_length,
        )

    del text_encoder
    del flux_pipeline.text_encoder

    return {
        "vae": vae,
        "vis": vis,
        "flow_decoder": flow_decoder,
        "scheduler": scheduler,
        "tokenizer": tokenizer,
        "flux_pipeline": flux_pipeline,
        "encoder_hidden_states": encoder_hidden_states.to(device, dtype=weight_dtype),
        "text_ids": text_ids.to(device),
        "repre_model": repre_model,
        "repre_processor": repre_processor,
    }

def compute_dino_representations(batch_im_a, batch_im_b, repre_model, repre_processor, device, weight_dtype):
    if repre_model is None or repre_processor is None:
        return None, None

    height, width = batch_im_a.shape[-2:]
    im_a_repre = repre_processor(
        batch_im_a,
        return_tensors="pt",
        do_resize=True,
        size={"height": height, "width": width},
        do_rescale=False,
        resample=Image.Resampling.BILINEAR,
    ).pixel_values.to(device, dtype=weight_dtype)
    im_b_repre = repre_processor(
        batch_im_b,
        return_tensors="pt",
        do_resize=True,
        size={"height": height, "width": width},
        do_rescale=False,
        resample=Image.Resampling.BILINEAR,
    ).pixel_values.to(device, dtype=weight_dtype)

    im_a_repre = repre_model(pixel_values=im_a_repre).last_hidden_state[:, 5:, :]
    im_b_repre = repre_model(pixel_values=im_b_repre).last_hidden_state[:, 5:, :]
    return im_a_repre, im_b_repre

class FreeMatching:
    """Single-pair dense matcher using the released Stage 1 or EMA Stage 2 weights."""

    def __init__(self, checkpoint, *, diffusion_model=None, representation_model=None,
                 device="cuda", dtype=torch.bfloat16, timestep=None):
        checkpoint = Path(checkpoint)
        config_path = checkpoint.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        self.device = torch.device(device)
        self.dtype = dtype
        self.timestep = config.get("timestep", 620) if timestep is None else timestep
        if not 0 <= self.timestep <= 1000:
            raise ValueError("timestep must be between 0 and 1000")
        self.height, self.width = config.get("height", 288), config.get("width", 512)
        args = SimpleNamespace(
            checkpoint=str(checkpoint),
            diffusion_model=diffusion_model or config.get("diffusion_model", "black-forest-labs/FLUX.2-klein-base-4B"),
            repre_model=representation_model or config.get("representation_model", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
            token_length=config.get("token_length", 2),
        )
        self.stack = build_model_stack(args, self.device, dtype, True, "flow")

    @torch.inference_mode()
    def __call__(self, source, target):
        """Inputs: same-shape Bx3xHxW RGB tensors in [0,1], H/W divisible by 16.

        Output grid maps each target pixel to its source coordinate (align_corners=True).
        """
        if source.shape != target.shape or source.ndim != 4 or source.shape[1] != 3:
            raise ValueError("source and target must have identical Bx3xHxW shapes")
        batch, _, height, width = source.shape
        if height % 16 or width % 16:
            raise ValueError("Image dimensions must be divisible by 16")
        a, b = source.to(self.device, self.dtype), target.to(self.device, self.dtype)
        stack = self.stack
        pipe = stack["flux_pipeline"]
        features = compute_dino_representations(a, b, stack["repre_model"], stack["repre_processor"], self.device, self.dtype)
        inputs = torch.cat([a, b], dim=0) * 2 - 1
        latent, image_ids = pipe.prepare_image_latents_parallel(inputs, 2, self.device, self.dtype)
        channels = latent.shape[-1]
        latent_a, latent_b = latent.chunk(2, dim=1)
        latent = torch.cat([latent_a.reshape(batch, -1, channels), latent_b.reshape(batch, -1, channels)], dim=1)
        prediction = stack["vis"](
            hidden_states=latent,
            timestep=torch.full((batch,), self.timestep, device=self.device) / 1000,
            guidance=None,
            encoder_hidden_states=repeat_batch_tensor(stack["encoder_hidden_states"], batch),
            txt_ids=repeat_batch_tensor(stack["text_ids"], batch),
            img_ids=repeat_batch_tensor(image_ids, batch),
            dino_repre=list(features),
        ).sample.chunk(2, dim=1)[1]
        output = decode_flux_output(pipe, prediction.to(self.dtype), height // 16, width // 16,
                                    output_decoder=stack["flow_decoder"]).float()
        grid = output[:, :2]
        confidence = ((output[:, 2:3] + 1) / 2).clamp(0, 1)
        return {"correspondence": grid, "confidence": confidence,
                "flow": correspondence_to_flow(grid), "warped": warp_source(a, grid)}
