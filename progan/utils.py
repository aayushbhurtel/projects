import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
	from torchvision.utils import make_grid, save_image
except Exception:
	make_grid = None
	save_image = None


def requires_grad(module: torch.nn.Module, flag: bool) -> None:
	for param in module.parameters():
		param.requires_grad = flag


def sample_latents(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
	return torch.randn(batch_size, latent_dim, device=device)


def denormalize_to_image(x: torch.Tensor) -> torch.Tensor:
	# From [-1, 1] to [0, 1]
	return torch.clamp((x + 1.0) * 0.5, 0.0, 1.0)


def save_image_grid(tensor: torch.Tensor, file_path: str, nrow: int = 8) -> None:
	if save_image is None or make_grid is None:
		return
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	grid = make_grid(denormalize_to_image(tensor), nrow=nrow)
	save_image(grid, file_path)


def compute_gradient_penalty(
	discriminator,
	real_samples: torch.Tensor,
	fake_samples: torch.Tensor,
	current_resolution: int,
	alpha: float,
	device: torch.device,
	lambda_gp: float = 10.0,
) -> torch.Tensor:
	batch_size = real_samples.size(0)
	# Random weight term for interpolation between real and fake samples
	epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
	interpolates = epsilon * real_samples + ((1.0 - epsilon) * fake_samples)
	interpolates.requires_grad_(True)

	d_interpolates = discriminator(interpolates, current_resolution=current_resolution, alpha=alpha)
	fake = torch.ones_like(d_interpolates, device=device)

	gradients = torch.autograd.grad(
		outputs=d_interpolates,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(batch_size, -1)
	gp = (((gradients.norm(2, dim=1) - 1.0) ** 2).mean()) * lambda_gp
	return gp