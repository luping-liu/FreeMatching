import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
            nn.ReLU(),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


class TinySD3Decoder(nn.Module):
    """
    Standalone Tiny AutoEncoder Decoder for SD3-style latents.
    Input:  (B, 32, H, W)
    Output: (B, 3, 8H, 8W)
    """

    def __init__(self):
        super().__init__()
        self.latent_channels = 32
        self.layers = nn.Sequential(
            Clamp(),
            conv(self.latent_channels, 64),
            nn.ReLU(),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            nn.Upsample(scale_factor=2),
            conv(64, 64, bias=False),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            nn.Upsample(scale_factor=2),
            conv(64, 64, bias=False),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
            nn.Upsample(scale_factor=2),
            conv(64, 64, bias=False),
            Block(64, 64),
            conv(64, 3),
        )

    def forward(self, x):
        return self.layers(x)

    def load_taesd_weights(self, path, map_location="cpu"):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        self.load_state_dict(state_dict)


class FlowDecoder(nn.Module):
    """
    Task decoder for dense normalized correspondence maps and masks.

    It keeps the same (B, 32, H, W) -> (B, 3, 8H, 8W) contract as TinySD3Decoder,
    but adds per-scale prediction heads and residual refinement at high resolution.
    The multi-scale heads give direct gradients before the final upsampling stages,
    which is useful for 1px-level coordinate errors.
    """

    def __init__(self, latent_channels=32, width=96, refine_residual_px_scale=4.0):
        super().__init__()
        self.refine_residual_px_scale = refine_residual_px_scale
        self.in_proj = nn.Sequential(Clamp(), conv(latent_channels, width), nn.SiLU(inplace=True))

        self.stage1 = nn.Sequential(Block(width, width), Block(width, width))
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), conv(width, width))
        self.flow_pred1 = nn.Conv2d(width, 2, 3, padding=1)
        self.mask_pred1 = nn.Conv2d(width, 1, 3, padding=1)

        self.stage2 = nn.Sequential(Block(width, width), Block(width, width))
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), conv(width, width))
        self.flow_pred2 = nn.Conv2d(width, 2, 3, padding=1)
        self.mask_pred2 = nn.Conv2d(width, 1, 3, padding=1)

        self.stage3 = nn.Sequential(Block(width, width), Block(width, width))
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), conv(width, width))
        self.flow_pred3 = nn.Conv2d(width, 2, 3, padding=1)
        self.mask_pred3 = nn.Conv2d(width, 1, 3, padding=1)

        self.flow_refine = nn.Sequential(
            Block(width + 2, width),
            Block(width, width),
            nn.Conv2d(width, 2, 3, padding=1),
        )
        self.mask_refine = nn.Sequential(
            Block(width + 1, width // 2),
            Block(width // 2, width // 2),
            nn.Conv2d(width // 2, 1, 3, padding=1),
        )

        self._init_output_heads()

    def _init_output_heads(self):
        for module in (
            self.flow_pred1,
            self.flow_pred2,
            self.flow_pred3,
            self.mask_pred1,
            self.mask_pred2,
            self.mask_pred3,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(module.bias)
        for module in (self.flow_refine[-1], self.mask_refine[-1]):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.in_proj(x)

        x = self.stage1(x)
        x = self.up1(x)
        flow = self.flow_pred1(x)
        mask = self.mask_pred1(x)

        x = self.stage2(x)
        x = self.up2(x)
        flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) + self.flow_pred2(x)
        mask = F.interpolate(mask, scale_factor=2, mode="bilinear", align_corners=True) + self.mask_pred2(x)

        x = self.stage3(x)
        x = self.up3(x)
        flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) + self.flow_pred3(x)
        mask = F.interpolate(mask, scale_factor=2, mode="bilinear", align_corners=True) + self.mask_pred3(x)

        flow_delta = self.flow_refine(torch.cat([x, flow], dim=1))
        _, _, h, w = flow.shape
        flow_scale = torch.tensor(
            [
                2.0 * self.refine_residual_px_scale / max(w - 1, 1),
                2.0 * self.refine_residual_px_scale / max(h - 1, 1),
            ],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, 2, 1, 1)
        flow = flow + torch.tanh(flow_delta) * flow_scale
        mask = mask + self.mask_refine(torch.cat([x, mask], dim=1))
        return torch.cat([flow, mask], dim=1)


def build_output_decoder(kind: str):
    if kind == "vae":
        return None
    if kind == "tiny_sd3":
        return TinySD3Decoder()
    if kind == "flow":
        return FlowDecoder()
    raise ValueError(f"Unknown output decoder: {kind}")


def flux_tokens_to_latents(flux_pipeline, image_latents, h_flux: int, w_flux: int):
    batch_size, total_seq_len, channels = image_latents.shape
    seq_len_per_img = h_flux * w_flux
    num_images_per_sample = total_seq_len // seq_len_per_img

    latents = image_latents.view(batch_size, num_images_per_sample, seq_len_per_img, channels)
    latents = latents.reshape(-1, seq_len_per_img, channels)
    latents = latents.permute(0, 2, 1).reshape(-1, channels, h_flux, w_flux)

    bn_mean = flux_pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
    bn_std = torch.sqrt(
        flux_pipeline.vae.bn.running_var.view(1, -1, 1, 1) + flux_pipeline.vae.config.batch_norm_eps
    ).to(latents.device, latents.dtype)
    latents = latents * bn_std + bn_mean
    return flux_pipeline._unpatchify_latents(latents)


def decode_flux_output(flux_pipeline, image_latents, h_flux: int, w_flux: int, output_decoder=None):
    if output_decoder is None:
        return flux_pipeline.decode_image_latents_parallel(image_latents, h_flux, w_flux)
    latents = flux_tokens_to_latents(flux_pipeline, image_latents, h_flux, w_flux)
    return output_decoder(latents)


class FluxFlowModel(nn.Module):
    """
    Single trainable module for DeepSpeed/Accelerate.

    It wraps the DiT (`vis`) and optional output decoder so distributed training only
    prepares one model object. The FLUX VAE/pipeline stays outside and is passed in
    only for deterministic token-to-latent conversion.
    """

    def __init__(self, vis, output_decoder=None):
        super().__init__()
        self.vis = vis
        self.output_decoder = output_decoder

    def forward(
        self,
        *,
        hidden_states,
        timestep,
        encoder_hidden_states,
        txt_ids,
        img_ids,
        dino_repre,
        flux_pipeline,
        h_flux: int,
        w_flux: int,
        output_index: int = 1,
        train_vis: bool = True,
    ):
        vis_context = torch.enable_grad() if train_vis else torch.no_grad()
        with vis_context:
            model_pred = self.vis(
                hidden_states=hidden_states,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=encoder_hidden_states,
                txt_ids=txt_ids,
                img_ids=img_ids,
                dino_repre=dino_repre,
            ).sample

        model_pred = model_pred.chunk(2, dim=1)[output_index]
        return decode_flux_output(
            flux_pipeline,
            model_pred.to(hidden_states.dtype),
            h_flux,
            w_flux,
            output_decoder=self.output_decoder,
        )
