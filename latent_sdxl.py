from typing import Any, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor,
)
from tqdm import tqdm

####### Factory #######
__SOLVER__ = {}


def register_solver(name: str):
    def wrapper(cls):
        if __SOLVER__.get(name, None) is not None:
            raise ValueError(f"Solver {name} already registered.")
        __SOLVER__[name] = cls
        return cls

    return wrapper


def get_solver(name: str, **kwargs):
    if name not in __SOLVER__:
        raise ValueError(f"Solver {name} does not exist.")
    return __SOLVER__[name](**kwargs)


########################


class SDXL:
    def __init__(
        self,
        solver_config: dict,
        pipeline: StableDiffusionXLPipeline = None,
        vae: AutoencoderKL = None,
        scheduler: DDIMScheduler = None,
        model_key: str = "stabilityai/stable-diffusion-xl-base-1.0",
        dtype=torch.float16,
        device="cuda",
    ):
        self.device = device
        if pipeline is None:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_key, torch_dtype=dtype
            ).to(device)
        else:
            pipe = pipeline
        self.dtype = dtype

        # Avoid overflow in float16
        if vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype
            ).to(device)
        else:
            self.vae = vae
        self.tokenizer_1 = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_enc_1 = pipe.text_encoder
        self.text_enc_2 = pipe.text_encoder_2
        self.unet = pipe.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.default_sample_size = self.unet.config.sample_size

        if scheduler is None:
            self.scheduler = DDIMScheduler.from_pretrained(
                model_key, subfolder="scheduler"
            )
        else:
            self.scheduler = scheduler
        self.total_alphas = self.scheduler.alphas_cumprod.clone()
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(solver_config.num_sampling, device=device)
        self.skip = N_ts // solver_config.num_sampling

        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(device)
        self.scheduler.alphas_cumprod = torch.cat(
            [torch.tensor([1.0]), self.scheduler.alphas_cumprod]
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.sample(*args, **kwargs)

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def _text_embed(self, prompt, tokenizer, text_enc, clip_skip):
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_enc(
            text_input_ids.to(self.device), output_hidden_states=True
        )
        pool_prompt_embeds = prompt_embeds[0]
        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]
        return prompt_embeds, pool_prompt_embeds

    @torch.no_grad()
    def get_text_embed(
        self,
        null_prompt_1,
        prompt_1,
        null_prompt_2=None,
        prompt_2=None,
        clip_skip=None,
    ):
        prompt_1 = [prompt_1] if isinstance(prompt_1, str) else prompt_1
        null_prompt_1 = (
            [null_prompt_1] if isinstance(null_prompt_1, str) else null_prompt_1
        )

        prompt_embed_1, pool_prompt_embed = self._text_embed(
            prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip
        )
        if prompt_2 is None:
            prompt_embed = [prompt_embed_1]
        else:
            prompt_embed_2, pool_prompt_embed = self._text_embed(
                prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip
            )
            prompt_embed = [prompt_embed_1, prompt_embed_2]

        null_embed_1, pool_null_embed = self._text_embed(
            null_prompt_1, self.tokenizer_1, self.text_enc_1, clip_skip
        )
        if null_prompt_2 is None:
            null_embed = [null_embed_1]
        else:
            null_embed_2, pool_null_embed = self._text_embed(
                null_prompt_2, self.tokenizer_2, self.text_enc_2, clip_skip
            )
            null_embed = [null_embed_1, null_embed_2]

        null_prompt_embeds = torch.concat(null_embed, dim=-1)
        prompt_embeds = torch.concat(prompt_embed, dim=-1)

        return null_prompt_embeds, prompt_embeds, pool_null_embed, pool_prompt_embed

    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
            (
                AttnProcessor2_0,
                XFormersAttnProcessor,
                LoRAXFormersAttnProcessor,
                LoRAAttnProcessor2_0,
            ),
        )
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, zt):
        needs_upcasting = (
            self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        )
        if needs_upcasting:
            self.upcast_vae()
            zt = zt.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
        image = self.vae.decode(zt / self.vae.config.scaling_factor).sample.float()
        return image.permute(0, 2, 3, 1)

    def predict_noise(self, zt, t, uc, c, added_cond_kwargs):
        t_in = t.unsqueeze(0)
        if uc is None:
            noise_c = self.unet(
                zt, t_in, encoder_hidden_states=c, added_cond_kwargs=added_cond_kwargs
            )["sample"]
            noise_uc = noise_c
        elif c is None:
            noise_uc = self.unet(
                zt, t_in, encoder_hidden_states=uc, added_cond_kwargs=added_cond_kwargs
            )["sample"]
            noise_c = noise_uc
        else:
            c_embed = torch.cat([uc, c], dim=0)
            z_in = torch.cat([zt] * 2)
            t_in = torch.cat([t_in] * 2)
            noise_pred = self.unet(
                z_in,
                t_in,
                encoder_hidden_states=c_embed,
                added_cond_kwargs=added_cond_kwargs,
            )["sample"]
            noise_uc, noise_c = noise_pred.chunk(2)

        return noise_uc, noise_c

    def _get_add_time_ids(
        self,
        original_size,
        crops_coords_top_left,
        target_size,
        dtype,
        text_encoder_projection_dim,
    ):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        passed_add_embed_dim = (
            self.unet.config.addition_time_embed_dim * len(add_time_ids)
            + text_encoder_projection_dim
        )
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features
        assert expected_add_embed_dim == passed_add_embed_dim, (
            f"Model expects an added time embedding vector of length {expected_add_embed_dim}, "
            f"but a vector of {passed_add_embed_dim} was created."
        )
        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample(
        self,
        zt,
        prompt1=["", ""],
        prompt2=["", ""],
        cfg_guidance: float = 5.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        (
            null_prompt_embeds,
            prompt_embeds,
            pool_null_embed,
            pool_prompt_embed,
        ) = self.get_text_embed(
            prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip
        )

        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            "text_embeds": add_text_embeds.to(self.device),
            "time_ids": add_time_ids.to(self.device),
        }

        zt = self.reverse_process(
            zt,
            null_prompt_embeds,
            prompt_embeds,
            cfg_guidance,
            add_cond_kwargs,
            target_size,
            **kwargs,
        )
        return zt

    @torch.autocast(device_type="cuda", dtype=torch.float16)
    def sample_latent(
        self,
        z0,
        prompt1=["", ""],
        prompt2=["", ""],
        cfg_guidance: float = 5.0,
        original_size: Optional[Tuple[int, int]] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Optional[Tuple[int, int]] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        clip_skip: Optional[int] = None,
        inversion_callback=None,
        **kwargs,
    ):
        height = self.default_sample_size * self.vae_scale_factor
        width = self.default_sample_size * self.vae_scale_factor
        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        (
            null_prompt_embeds,
            prompt_embeds,
            pool_null_embed,
            pool_prompt_embed,
        ) = self.get_text_embed(
            prompt1[0], prompt1[1], prompt2[0], prompt2[1], clip_skip
        )

        add_text_embeds = pool_prompt_embed
        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
        )

        if negative_original_size is not None and negative_target_size is not None:
            negative_add_time_ids = self._get_add_time_ids(
                negative_original_size,
                negative_crops_coords_top_left,
                negative_target_size,
                dtype=prompt_embeds.dtype,
                text_encoder_projection_dim=int(pool_prompt_embed.shape[-1]),
            )
        else:
            negative_add_time_ids = add_time_ids
        negative_text_embeds = pool_null_embed

        if cfg_guidance != 0.0 and cfg_guidance != 1.0:
            add_text_embeds = torch.cat([negative_text_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

        add_cond_kwargs = {
            "text_embeds": add_text_embeds.to(self.device),
            "time_ids": add_time_ids.to(self.device),
        }

        zt = self.inversion(
            z0,
            null_prompt_embeds,
            prompt_embeds,
            cfg_guidance=cfg_guidance,
            add_cond_kwargs=add_cond_kwargs,
            inversion_callback=inversion_callback,
        )
        return zt

    def inversion(
        self, z0, uc, c, cfg_guidance, add_cond_kwargs, inversion_callback=None
    ):
        if cfg_guidance == 0.0 or cfg_guidance == 1.0:
            add_cond_kwargs["text_embeds"] = add_cond_kwargs["text_embeds"][
                -1
            ].unsqueeze(0)
            add_cond_kwargs["time_ids"] = add_cond_kwargs["time_ids"][-1].unsqueeze(0)

        zt = z0.clone().to(self.device)
        pbar = tqdm(reversed(self.scheduler.timesteps), desc="DDIM inversion")

        all_latents = {}
        for step_idx, t in enumerate(pbar):
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(zt, t, uc, c, add_cond_kwargs)
                if inversion_callback is not None:
                    inversion_callback(step_idx=step_idx, timestep=t)
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at_prev).sqrt() * noise_pred) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1 - at).sqrt() * noise_pred
            all_latents[str(t.item())] = zt

        return all_latents

    def reverse_process(self, *args, **kwargs):
        raise NotImplementedError


@register_solver("ddim")
class BaseDDIM(SDXL):
    def reverse_process(
        self,
        zt,
        null_prompt_embeds,
        prompt_embeds,
        cfg_guidance,
        add_cond_kwargs,
        shape=(1024, 1024),
        callback_fn=None,
        **kwargs,
    ):
        pbar = tqdm(self.scheduler.timesteps.int(), desc="SDXL")
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(
                    zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs
                )
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_pred

            if callback_fn is not None:
                callback_kwargs = {
                    "z0t": z0t.detach(),
                    "zt": zt.detach(),
                    "decode": self.decode,
                }
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        return z0t


@register_solver("ddim_cfg++")
class BaseDDIMCFGpp(SDXL):
    def reverse_process(
        self,
        zt,
        null_prompt_embeds,
        prompt_embeds,
        cfg_guidance,
        add_cond_kwargs,
        shape=(1024, 1024),
        callback_fn=None,
        **kwargs,
    ):
        pbar = tqdm(self.scheduler.timesteps.int(), desc="SDXL")
        for step, t in enumerate(pbar):
            next_t = t - self.skip
            at = self.scheduler.alphas_cumprod[t]
            at_next = self.scheduler.alphas_cumprod[next_t]

            with torch.no_grad():
                noise_uc, noise_c = self.predict_noise(
                    zt, t, null_prompt_embeds, prompt_embeds, add_cond_kwargs
                )
                noise_pred = noise_uc + cfg_guidance * (noise_c - noise_uc)

            z0t = (zt - (1 - at).sqrt() * noise_pred) / at.sqrt()
            zt = at_next.sqrt() * z0t + (1 - at_next).sqrt() * noise_uc

            if callback_fn is not None:
                callback_kwargs = {
                    "z0t": z0t.detach(),
                    "zt": zt.detach(),
                    "decode": self.decode,
                }
                callback_kwargs = callback_fn(step, t, callback_kwargs)
                z0t = callback_kwargs["z0t"]
                zt = callback_kwargs["zt"]

        return z0t
