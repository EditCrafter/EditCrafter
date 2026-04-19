import os

import numpy as np
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def seed_everything(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_views(
    img_height,
    img_width,
    window_size=64,
    stride=8,
    latent_space=True,
    circular_padding=False,
):
    if latent_space:
        img_height /= 8
        img_width /= 8
    num_blocks_height = (
        (img_height - window_size) // stride + 1 if img_height > window_size else 1
    )
    if circular_padding:
        num_blocks_width = img_width // stride if img_width > window_size else 1
    else:
        num_blocks_width = (
            (img_width - window_size) // stride + 1 if img_width > window_size else 1
        )
    total_num_blocks = int(num_blocks_height * num_blocks_width)
    views = []
    for i in range(total_num_blocks):
        h_start = int((i // num_blocks_width) * stride)
        h_end = h_start + window_size
        w_start = int((i % num_blocks_width) * stride)
        w_end = w_start + window_size
        views.append((h_start, h_end, w_start, w_end))
    return views


class DDIM_inversion:
    def __init__(self, model, num_inference_steps=100):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.num_inference_steps = num_inference_steps
        self.model.scheduler.set_timesteps(self.num_inference_steps)
        self.prompt = None
        self.context = None
        self.skip = 1000 // num_inference_steps
        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(model.device)
        self.scheduler.alphas_cumprod = torch.cat(
            [torch.tensor([1.0]), self.scheduler.alphas_cumprod]
        )

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def latent2image(self, latents, return_type="np"):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)["sample"]
        if return_type == "np":
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image) / 127.5 - 1
                image = (
                    image.permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self.model.device, dtype=self.model.vae.dtype)
                )
                latents = self.model.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt, text_encoder=None):
        uncond_input = self.model.tokenizer(
            [""],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt",
        )
        if text_encoder is None:
            uncond_embeddings = self.model.text_encoder(
                uncond_input.input_ids.to(self.model.device)
            )[0]
        else:
            uncond_embeddings = text_encoder(
                uncond_input.input_ids.to(self.model.device)
            )[0]

        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        if text_encoder is None:
            text_embeddings = self.model.text_encoder(
                text_input.input_ids.to(self.model.device)
            )[0]
        else:
            text_embeddings = text_encoder(text_input.input_ids.to(self.model.device))[
                0
            ]

        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
        return self.context

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = (
            min(
                timestep
                - self.scheduler.config.num_train_timesteps
                // self.scheduler.num_inference_steps,
                999,
            ),
            timestep,
        )
        alpha_prod_t = (
            self.scheduler.alphas_cumprod[timestep]
            if timestep >= 0
            else self.scheduler.final_alpha_cumprod
        )
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (
            sample - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = (
            alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        )
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)[
            "sample"
        ]
        return noise_pred

    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at

    @torch.no_grad()
    def ddim_loop(self, latent):
        assert self.context is not None
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = {"0": latent}
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_inference_steps), desc="DDIM Inverting"):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent[str(t.item())] = latent
        return all_latent

    @torch.no_grad()
    def ddim_cfgpp_loop(self, latent, guidance_scale=0.02):
        assert self.context is not None
        all_latent = {"0": latent}
        zt = latent.clone().detach()

        for i in tqdm(range(self.num_inference_steps), desc="DDIM CFG++ Inverting"):
            t = self.model.scheduler.timesteps[
                len(self.model.scheduler.timesteps) - i - 1
            ]
            at = self.alpha(t)
            at_prev = self.alpha(t - self.skip)

            noise_pred = self.get_noise_pred_single(
                torch.cat([zt] * 2), t, self.context
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

            z0t = (zt - (1 - at_prev).sqrt() * noise_pred_uncond) / at_prev.sqrt()
            zt = at.sqrt() * z0t + (1 - at).sqrt() * noise_pred
            all_latent[str(t.item())] = zt
        return all_latent

    @torch.no_grad()
    def ddim_invert(self, image, guidance_scale, is_cfgpp=False):
        latent = self.image2latent(image)
        if is_cfgpp:
            ddim_latents = self.ddim_cfgpp_loop(latent, guidance_scale=guidance_scale)
        else:
            ddim_latents = self.ddim_loop(latent)
        return ddim_latents


class DDIM_tiled_inversion(DDIM_inversion):
    def __init__(self, model, num_inference_steps=50):
        super().__init__(model, num_inference_steps)
        self.img_height = 0
        self.img_width = 0

    def make_patches(self, image, views, latent_space=False):
        patches = []
        for view in views:
            h_start, h_end, w_start, w_end = view
            if latent_space:
                h_start, h_end, w_start, w_end = (
                    h_start // 8,
                    h_end // 8,
                    w_start // 8,
                    w_end // 8,
                )
            patch = image[h_start:h_end, w_start:w_end]
            patches.append(patch)
        return patches

    def join_patches(self, patches, views, latent_space=True):
        height, width = 0, 0
        for view in views:
            h_start, h_end, w_start, w_end = view
            height = max(height, h_end)
            width = max(width, w_end)

        if latent_space:
            height, width = height // 8, width // 8

        if latent_space:
            batch_size = patches[0].shape[0]
            image = torch.zeros(batch_size, 4, height, width).to(
                self.model.device, dtype=torch.float32
            )
        else:
            image = np.zeros((height, width, 3), dtype=np.uint8)

        for patch, (h_start, h_end, w_start, w_end) in zip(patches, views):
            if latent_space:
                image[:, :, h_start // 8 : h_end // 8, w_start // 8 : w_end // 8] = (
                    patch
                )
            else:
                image[h_start:h_end, w_start:w_end, :] = patch

        return image

    @torch.no_grad()
    def ddim_tile_invert(
        self,
        image,
        guidance_scale=0.02,
        inversion_strength=1.0,
        is_cfgpp=True,
        window_size=512,
        **kwargs,
    ):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
            image = np.array(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        self.img_height, self.img_width = image.shape[0], image.shape[1]
        views = get_views(
            self.img_height,
            self.img_width,
            window_size=window_size,
            stride=window_size,
            latent_space=False,
        )
        original_img_patch_list = self.make_patches(image, views)

        inv_time_step_idx = int(self.num_inference_steps * inversion_strength)
        inv_time_step = str(self.model.scheduler.timesteps[-inv_time_step_idx].numpy())
        print(
            f"Inversion strength is {inversion_strength}, invert until time step {inv_time_step}"
        )

        latent_patch_list = []
        for i, view in enumerate(views):
            print(f"Inverting patch {i + 1}/{len(views)}...")
            inv_latent_patch = self.ddim_invert(
                original_img_patch_list[i],
                guidance_scale=guidance_scale,
                is_cfgpp=is_cfgpp,
            )[inv_time_step]
            latent_patch_list.append(inv_latent_patch)

        inv_latent = self.join_patches(latent_patch_list, views, latent_space=True)
        print("Inverted latent shape:", inv_latent.shape)
        return inv_latent, latent_patch_list
