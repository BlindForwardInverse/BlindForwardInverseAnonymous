import os
import pickle

import dnnlib
import numpy as np
import PIL
import soundfile as sf
import torch
import torch.nn as nn
import torchvision.utils as tvu
import torch.nn.functional as F
from einops import rearrange
from svd_replacement import Deblurring, Denoising, Inpainting, SuperResolution
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AFHQDataset

opj = os.path.join


def generate_image_grid(
    network_pkl,
    dest_path,
    seed=0,
    gridw=8,
    gridh=8,
    device=torch.device("cuda"),
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
    S_churn=40,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    batch_size = gridw * gridh
    torch.manual_seed(seed)

    # Load network.
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)["ema"].to(device)

    # Pick latents and labels.
    print(f"Generating {batch_size} images...")
    class_labels = None
    if net.label_dim:
        class_labels = torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)
        ]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    # --------------------------------------------------------------------
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    dataset = AFHQDataset()
    image_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1)
    num_test = 1

    for idx, y in zip(range(num_test), iter(image_loader)):
        y = y.to(device)
        shape = y.shape
        print(shape)
        operator = inpaint
        y = operator(y)
        latents = torch.randn(
            [batch_size, net.img_channels, net.img_resolution, net.img_resolution],
            device=device,
            requires_grad=True,
        )
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in tqdm(
            list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit="step"
        ):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = (
                min(S_churn / num_steps, np.sqrt(2) - 1)
                if S_min <= t_cur <= S_max
                else 0
            )
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (
                t_hat**2 - t_cur**2
            ).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            # denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
            d_uncond, d_cond = recon_guid(net, x_hat, t_hat, class_labels, operator, y)
            print(torch.linalg.norm(d_uncond).item(), torch.linalg.norm(d_cond).item())

            d_cur = d_uncond - d_cond
            # d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                # denoised = net(x_next, t_next, class_labels).to(torch.float64)
                # d_prime = (x_next - denoised) / t_next
                d_uncond, d_cond = recon_guid(
                    net, x_next, t_next, class_labels, operator, y
                )
                d_prime = d_uncond - d_cond
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        # Save image grid.
        print(f'Saving image grid to "{dest_path}"...')
        image = (x_next * 127.5 * 2+ 128).clip(0, 255).to(torch.uint8)
        image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
        image = image.reshape(
            gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels
        )
        image = image.cpu().numpy()
        PIL.Image.fromarray(image, "RGB").save(dest_path)

        image = (y * 127.5 + 128).clip(0, 255).to(torch.uint8)
        image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
        image = image.reshape(
            gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels
        )
        image = image.cpu().numpy()
        PIL.Image.fromarray(image, "RGB").save(dest_path[:-4]+"_gt.png")
    print("Done.")

def recon_guid(net, x_t, t, class_labels=None, operator=None, y=None):
    denoised = net(x_t, t, class_labels).to(torch.float32)
    difference = y - operator(denoised)
    norm = torch.sum(torch.square(difference)) * t
    
#     l2_loss = F.mse_loss(y, operator(denoised), reduction='sum') * t
    
    gradient = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
#     weight = calc_weight(difference)
    weight = 1.0
    guid = -weight * gradient

    d_uncond = (x_t - denoised) / t
    d_cond = guid
    return d_uncond, d_cond


def calc_weight(difference):
    weight = 0.3
    return weight

def inpaint(x):
    size = x.shape[-1]
    idx = int(0.7 * size)
    mask = torch.ones_like(x)
    mask[..., idx:] = 0
    add_mask = torch.zeros_like(x)
    add_mask[..., idx:] = -1
    x = mask * x + add_mask
    return x

def identity(x):
    return x


def get_missing(img_size):
    missing_r = torch.randperm(img_size**2)[: img_size**2 // 2].long() * 3

    missing_g = missing_r + 1
    missing_b = missing_g + 1
    missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
    return missing


def inpainting(x, net):
    device = x.get_device()

    img_size = net.img_resolution  # 256
    img_channels = net.img_channels  # 3
    print(img_size)
    print(img_channels)

    # H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, self.device)

    operation = Inpainting(img_channels, img_size, get_missing(), device)
    # operation = torch.eye(x.shape) ##
    return operation


def save_to_image(what, dest_path, net, gridh=8, gridw=8):
    print(f'Saving image grid to "{dest_path}"...')
    image = (what * 127.5 + 128).clip(0, 255).to(torch.uint8)
    image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
    image = image.reshape(
        gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels
    )
    image = image.cpu().numpy()
    PIL.Image.fromarray(image, "RGB").save(dest_path)
    print("Done.")


# ----------------------------------------------------------------------------


def main():
    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
    generate_image_grid(
        f"{model_root}/edm-afhqv2-64x64-uncond-vp.pkl", "/ssd3/doyo/dps_out/dps_afhqv2-64x64.png", num_steps=64
    )  # FID = 1.96, NFE = 79


if __name__ == "__main__":
    main()
