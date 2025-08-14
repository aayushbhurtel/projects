from typing import Dict, List

import torch
import torch.nn as nn

from .generator import Generator
from .discriminator import Discriminator


class ProGAN(nn.Module):
	def __init__(self, latent_dim: int = 512, max_resolution: int = 128):
		super().__init__()
		self.latent_dim = latent_dim
		self.max_resolution = max_resolution
		self.generator = Generator(latent_dim=latent_dim, max_resolution=max_resolution)
		self.discriminator = Discriminator(max_resolution=max_resolution)

	@torch.inference_mode()
	def generate(self, z: torch.Tensor, current_resolution: int, alpha: float = 1.0) -> torch.Tensor:
		return self.generator(z, current_resolution=current_resolution, alpha=alpha)