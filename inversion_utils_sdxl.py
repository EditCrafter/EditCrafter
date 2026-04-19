import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


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


class DDIM_tiled_inversion:
    def __init__(self, model, solver_config):
        self.model = model
        self.img_height = 0
        self.img_width = 0
        self.num_inference_steps = solver_config.num_sampling

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
    def image2latent(self, image, vae_tiling=False):
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
                if vae_tiling:
                    latents = self.model.vae.tiled_encode(image)["latent_dist"].mean
                else:
                    latents = self.model.vae.encode(image)["latent_dist"].mean
                latents = latents * 0.18215
        return latents

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
    def ddim_tile_invert(
        self,
        image,
        prompts,
        guidance_scale=0.02,
        inversion_strength=1.0,
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
        inv_time_step = str(
            self.model.scheduler.timesteps[-inv_time_step_idx].cpu().numpy()
        )
        print(
            f"Inversion strength is {inversion_strength}, invert until time step {inv_time_step}"
        )

        latent_patch_list = []
        for i, view in enumerate(views):
            print(f"Inverting patch {i + 1}/{len(views)}...")
            inv_latent_patch = self.model.sample_latent(
                self.image2latent(original_img_patch_list[i]),
                prompt1=prompts,
                prompt2=prompts,
                cfg_guidance=guidance_scale,
            )[inv_time_step]
            latent_patch_list.append(inv_latent_patch)

        inv_latent = self.join_patches(latent_patch_list, views, latent_space=True)
        print("Inverted latent shape:", inv_latent.shape)
        return inv_latent, latent_patch_list

    @torch.no_grad()
    def ddim_tile_reverse_step(
        self,
        latents,
        prompts,
        guidance_scale=0.02,
        inversion_strength=1.0,
        window_size=512,
        **kwargs,
    ):
        if self.img_height == 0 or self.img_width == 0:
            raise ValueError(
                "Image height and width must be set. Please run ddim_tile_invert() first."
            )

        views = get_views(
            self.img_height,
            self.img_width,
            window_size=window_size,
            stride=window_size,
            latent_space=False,
        )

        inv_time_step_idx = int(self.num_inference_steps * inversion_strength)
        inv_time_step = self.model.scheduler.timesteps[-inv_time_step_idx].cpu().numpy()
        print(f"Reverse process starts from time step {inv_time_step}")

        img_patch_list = []
        for i, view in enumerate(views):
            print(f"Latent patch to image {i + 1}/{len(views)}...")
            latent_patch = self.model.sample(
                latents[i],
                prompt1=prompts,
                prompt2=prompts,
                cfg_guidance=guidance_scale,
            )
            latent_patch = self.latent2image(latent_patch)
            img_patch_list.append(latent_patch)

        recon_img = self.join_patches(img_patch_list, views, latent_space=False)
        print("Reconstructed image shape:", recon_img.shape)
        return recon_img, img_patch_list
