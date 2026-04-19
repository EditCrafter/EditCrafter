import argparse
import copy
import gc
import glob
import math
import os
from typing import Optional

import scipy
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from munch import munchify
from omegaconf import OmegaConf
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from free_lunch_utils import register_free_crossattn_upblock2d, register_free_upblock2d
from inversion_utils_sdxl import DDIM_tiled_inversion
from latent_sdxl import get_solver
from model import ReDilateConvProcessor, inflate_kernels
from sync_tiled_decode import apply_sync_tiled_decode, apply_tiled_processors

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-guided high-resolution image editing with EditCrafter (SDXL)."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="An image path to be edited.",
    )
    parser.add_argument(
        "--inversion_prompt",
        type=str,
        default="",
        help="A prompt used for DDIM inversion.",
    )
    parser.add_argument(
        "--editing_prompt",
        type=str,
        required=True,
        help="A prompt that describes the desired edit.",
    )
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    parser.add_argument("--inversion_strength", type=float, default=1.0)
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=23, help="A seed for reproducible inference."
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of inference steps."
    )
    parser.add_argument("--config", type=str, default="./configs/sdxl_2048x2048.yaml")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument("--disable_freeu", action="store_true", default=False)
    parser.add_argument("--vae_tiling", action="store_true", help="Enable VAE tiling.")
    parser.add_argument(
        "--guidance_type",
        type=str,
        default="cfgpp",
        choices=["cfg", "cfgpp"],
        help="Guidance type: cfg or cfgpp.",
    )
    parser.add_argument("--inversion_method", type=str, default="ddim_cfg++")
    args = parser.parse_args()
    return args


def pipeline_processor(
    self,
    ndcfg_tau=0,
    dilate_tau=0,
    inflate_tau=0,
    sdedit_tau=0,
    dilate_settings=None,
    inflate_settings=None,
    ndcfg_dilate_settings=None,
    transform=None,
    progressive=False,
    is_cfgpp=False,
    num_inference_steps=50,
    guidance_scale=0.02,
):
    if is_cfgpp:
        self.skip = 1000 // num_inference_steps
        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod
        self.scheduler.alphas_cumprod = torch.cat(
            [torch.tensor([1.0]), self.scheduler.alphas_cumprod]
        )

    @torch.no_grad()
    def alpha(t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def forward(
        prompt=None,
        prompt_2=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = num_inference_steps,
        denoising_end: Optional[float] = None,
        guidance_scale: float = guidance_scale,
        negative_prompt=None,
        negative_prompt_2=None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 1.0,
        generator=None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback=None,
        callback_steps: int = 1,
        cross_attention_kwargs=None,
        guidance_rescale: float = 0.0,
        original_size=None,
        crops_coords_top_left=(0, 0),
        target_size=None,
        is_cfgpp=is_cfgpp,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = next(self.unet.parameters()).device
        do_classifier_free_guidance = guidance_scale > 1.0 or (
            guidance_scale > 0.0 and is_cfgpp
        )

        # Move text encoders to GPU for encoding, then offload to save VRAM
        _te_device = next(self.text_encoder.parameters()).device
        if _te_device != device:
            self.text_encoder.to(device)
            self.text_encoder_2.to(device)

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # Offload text encoders to free ~1.9 GB VRAM during denoising
        self.text_encoder.to("cpu")
        self.text_encoder_2.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=self.text_encoder_2.config.projection_dim,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        if (
            denoising_end is not None
            and type(denoising_end) == float
            and 0 < denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        unet_inflate, unet_inflate_vanilla = None, None
        if transform is not None:
            unet_inflate = copy.deepcopy(self.unet)
            if inflate_settings is not None:
                inflate_kernels(unet_inflate, inflate_settings, transform)

        if transform is not None and ndcfg_tau > 0:
            unet_inflate_vanilla = copy.deepcopy(self.unet)
            if inflate_settings is not None:
                inflate_kernels(unet_inflate_vanilla, inflate_settings, transform)

        if sdedit_tau is not None:
            timesteps = timesteps[sdedit_tau:]

        # Pre-split CFG embeddings for sequential batch=1 UNet passes (halves peak VRAM)
        if do_classifier_free_guidance:
            prompt_embeds_uncond, prompt_embeds_cond = prompt_embeds.chunk(2)
            add_text_embeds_uncond, add_text_embeds_cond = add_text_embeds.chunk(2)
            add_time_ids_uncond, add_time_ids_cond = add_time_ids.chunk(2)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = self.scheduler.scale_model_input(latents, t)

                unet = (
                    unet_inflate
                    if i < inflate_tau and transform is not None
                    else self.unet
                )
                backup_forwards = dict()
                for name, module in unet.named_modules():
                    if name in dilate_settings.keys():
                        backup_forwards[name] = module.forward
                        dilate = dilate_settings[name]
                        if progressive:
                            dilate = max(
                                math.ceil(dilate * ((dilate_tau - i) / dilate_tau)), 2
                            )
                        if i < inflate_tau and name in inflate_settings:
                            dilate = dilate / 2
                        module.forward = ReDilateConvProcessor(
                            module, dilate, mode="bilinear", activate=i < dilate_tau
                        )

                if do_classifier_free_guidance:
                    added_cond_kwargs_uncond = {
                        "text_embeds": add_text_embeds_uncond,
                        "time_ids": add_time_ids_uncond,
                    }
                    noise_pred_uncond = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_uncond,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs_uncond,
                        return_dict=False,
                    )[0]

                    added_cond_kwargs_cond = {
                        "text_embeds": add_text_embeds_cond,
                        "time_ids": add_time_ids_cond,
                    }
                    noise_pred_text = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_cond,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs_cond,
                        return_dict=False,
                    )[0]
                else:
                    added_cond_kwargs = {
                        "text_embeds": add_text_embeds,
                        "time_ids": add_time_ids,
                    }
                    noise_pred_uncond = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_text = noise_pred_uncond

                for name, module in unet.named_modules():
                    if name in backup_forwards.keys():
                        module.forward = backup_forwards[name]

                if i < ndcfg_tau:
                    latents = process_with_ndcfg_tau(
                        guidance_scale,
                        latents,
                        cross_attention_kwargs,
                        guidance_rescale,
                        is_cfgpp,
                        do_classifier_free_guidance,
                        extra_step_kwargs,
                        unet_inflate_vanilla,
                        i,
                        t,
                        latent_model_input,
                        noise_pred_uncond,
                        noise_pred_text,
                        ndcfg_tau,
                        inflate_tau,
                        progressive,
                        ndcfg_dilate_settings,
                        inflate_settings,
                        transform,
                        prompt_embeds_uncond=prompt_embeds_uncond,
                        added_cond_kwargs_uncond=added_cond_kwargs_uncond,
                    )
                else:
                    latents = process_without_ndcfg_tau(
                        guidance_scale,
                        latents,
                        guidance_rescale,
                        is_cfgpp,
                        do_classifier_free_guidance,
                        extra_step_kwargs,
                        i,
                        t,
                        noise_pred_uncond,
                        noise_pred_text,
                    )

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # Free UNet VRAM before VAE decode
        if unet_inflate is not None:
            del unet_inflate
        if unet_inflate_vanilla is not None:
            del unet_inflate_vanilla
        self.unet.to("cpu")
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()

        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
        else:
            image = latents
            self.unet.to(device)
            return StableDiffusionXLPipelineOutput(images=image)

        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        self.unet.to(device)

        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    def process_with_ndcfg_tau(
        guidance_scale,
        latents,
        cross_attention_kwargs,
        guidance_rescale,
        is_cfgpp,
        do_classifier_free_guidance,
        extra_step_kwargs,
        unet_inflate_vanilla,
        i,
        t,
        latent_model_input,
        noise_pred_uncond,
        noise_pred_text,
        ndcfg_tau,
        inflate_tau,
        progressive,
        ndcfg_dilate_settings,
        inflate_settings,
        transform,
        prompt_embeds_uncond=None,
        added_cond_kwargs_uncond=None,
    ):
        unet = (
            unet_inflate_vanilla
            if i < inflate_tau and transform is not None
            else self.unet
        )
        backup_forwards = dict()

        for name, module in unet.named_modules():
            if name in ndcfg_dilate_settings.keys():
                backup_forwards[name] = module.forward
                dilate = ndcfg_dilate_settings[name]
                if progressive:
                    dilate = max(math.ceil(dilate * ((ndcfg_tau - i) / ndcfg_tau)), 2)
                if i < inflate_tau and name in inflate_settings:
                    dilate = dilate / 2
                module.forward = ReDilateConvProcessor(
                    module, dilate, mode="bilinear", activate=i < ndcfg_tau
                )

        noise_pred_vanilla = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds_uncond,
            cross_attention_kwargs=cross_attention_kwargs,
            added_cond_kwargs=added_cond_kwargs_uncond,
            return_dict=False,
        )[0]

        for name, module in unet.named_modules():
            if name in backup_forwards.keys():
                module.forward = backup_forwards[name]

        if do_classifier_free_guidance:
            noise_pred = noise_pred_vanilla + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

        if is_cfgpp:
            latents = cfg_process(latents, t, noise_pred, noise_pred_vanilla)
        else:
            variance_noise = None
            results = self.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                variance_noise=variance_noise,
                return_dict=True,
            )
            latents = results.prev_sample

        return latents

    def cfg_process(zt, t, noise_pred, noise_pred_basis):
        at = alpha(t)
        at_next = alpha(t - self.skip)
        z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
        latents = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_pred_basis
        return latents

    def process_without_ndcfg_tau(
        guidance_scale,
        latents,
        guidance_rescale,
        is_cfgpp,
        do_classifier_free_guidance,
        extra_step_kwargs,
        i,
        t,
        noise_pred_uncond,
        noise_pred_text,
    ):
        if do_classifier_free_guidance:
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
            if guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                )

        if is_cfgpp:
            latents = cfg_process(latents, t, noise_pred, noise_pred_uncond)
        else:
            variance_noise = None
            results = self.scheduler.step(
                noise_pred,
                t,
                latents,
                **extra_step_kwargs,
                variance_noise=variance_noise,
                return_dict=True,
            )
            latents = results.prev_sample

        return latents

    return forward


def read_module_list(path):
    with open(path, "r") as f:
        module_list = f.readlines()
        module_list = [name.strip() for name in module_list]
    return module_list


def read_dilate_settings(path):
    dilate_settings = dict()
    with open(path, "r") as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            name, dilate = raw_line.split(":")
            dilate_settings[name] = float(dilate)
    return dilate_settings


def main():
    args = parse_args()
    logging_dir = os.path.join(args.logging_dir)
    config = OmegaConf.load(args.config)

    accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    if weight_dtype == torch.float32:
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            torch_dtype=weight_dtype,
        )
    elif weight_dtype == torch.float16:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=weight_dtype
        )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="unet",
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)

    if not args.disable_freeu:
        register_free_upblock2d(pipeline, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        register_free_crossattn_upblock2d(pipeline, b1=1.1, b2=1.2, s1=0.6, s2=0.4)

    if args.vae_tiling:
        pipeline.enable_vae_tiling()
        apply_sync_tiled_decode(pipeline.vae)
        apply_tiled_processors(pipeline.vae.decoder)

    dilate_settings = (
        read_dilate_settings(config.dilate_settings)
        if config.dilate_settings is not None
        else dict()
    )
    ndcfg_dilate_settings = (
        read_dilate_settings(config.ndcfg_dilate_settings)
        if config.ndcfg_dilate_settings is not None
        else dict()
    )
    inflate_settings = (
        read_module_list(config.disperse_settings)
        if config.disperse_settings is not None
        else list()
    )
    if config.disperse_transform is not None:
        transform = scipy.io.loadmat(config.disperse_transform)["R"]
        transform = torch.tensor(transform, device=accelerator.device)
    else:
        transform = None

    unet.eval()

    os.makedirs(os.path.join(logging_dir), exist_ok=True)
    total_num = len(glob.glob(os.path.join(logging_dir, "*.jpg"))) - 1

    if os.path.isfile(args.editing_prompt):
        with open(args.editing_prompt, "r") as f:
            editing_prompt = f.readlines()
            editing_prompt = [line.strip() for line in editing_prompt]
    else:
        editing_prompt = [args.editing_prompt]

    inference_batch_size = config.inference_batch_size
    num_batches = math.ceil(len(editing_prompt) / inference_batch_size)

    img = Image.open(args.img_path).convert("RGB")

    solver_config = munchify({"num_sampling": 50})
    solver = get_solver(
        args.inversion_method,
        solver_config=solver_config,
        pipeline=pipeline,
        vae=vae,
        scheduler=noise_scheduler,
        device=accelerator.device,
    )

    inversion_prompt = args.inversion_prompt

    ddim_inv = DDIM_tiled_inversion(solver, solver_config)
    inv_latent, _ = ddim_inv.ddim_tile_invert(
        img,
        prompts=["", inversion_prompt],
        guidance_scale=0.0,
        inversion_strength=args.inversion_strength,
        window_size=1024,
    )

    inv_latent = inv_latent.to(accelerator.device, dtype=weight_dtype)
    torch.cuda.empty_cache()

    for i in range(num_batches):
        output_prompts = editing_prompt[
            i * inference_batch_size : min(
                (i + 1) * inference_batch_size, len(editing_prompt)
            )
        ]

        for n in range(config.num_iters_per_prompt):
            seed = args.seed + n
            set_seed(seed)

            pipeline.enable_vae_tiling()
            pipeline.forward = pipeline_processor(
                pipeline,
                ndcfg_tau=config.ndcfg_tau,
                dilate_tau=config.dilate_tau,
                inflate_tau=config.inflate_tau,
                dilate_settings=dilate_settings,
                inflate_settings=inflate_settings,
                ndcfg_dilate_settings=ndcfg_dilate_settings,
                transform=transform,
                progressive=config.progressive,
                is_cfgpp=(args.guidance_type == "cfgpp"),
                num_inference_steps=config.num_inference_steps,
                guidance_scale=args.guidance_scale,
            )
            images = pipeline.forward(
                output_prompts,
                output_prompts,
                negative_prompt=[""],
                negative_prompt_2=[""],
                generator=None,
                latents=inv_latent,
                num_inference_steps=config.num_inference_steps,
                height=config.pixel_height,
                width=config.pixel_width,
            ).images

            for image, prompt in zip(images, output_prompts):
                total_num = total_num + 1
                img_path = os.path.join(
                    logging_dir, f"{total_num}_{prompt[:200]}_seed{seed}.jpg"
                )
                image.save(img_path)
                with open(os.path.join(logging_dir, f"{total_num}.txt"), "w") as f:
                    f.writelines(
                        [
                            f"original_img_path: {args.img_path}\n",
                            f"edit_prompt: {prompt}\n",
                            f"inversion_prompt: {inversion_prompt}\n",
                            f"ndcfg_tau: {config.ndcfg_tau}\n",
                            f"dilate_tau: {config.dilate_tau}\n",
                            f"inflate_tau: {config.inflate_tau}\n",
                            f"guidance_scale: {args.guidance_scale}\n",
                            f"is_cfgpp: {(args.guidance_type == 'cfgpp')}\n",
                            f"inversion_type: tiled\n",
                        ]
                    )


if __name__ == "__main__":
    main()
