from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import EqualizedLRLinear, EqualizedLRConv2d, GeneratorConvBlock, PixelNorm


def default_channel_map(max_resolution: int) -> Dict[int, int]:
	# Resolution to channel count mapping, following common ProGAN practice
	full_map = {
		4: 512,
		8: 512,
		16: 512,
		32: 256,
		64: 128,
		128: 64,
		256: 32,
		512: 16,
		1024: 8,
	}
	allowed = [r for r in full_map.keys() if r <= max_resolution]
	return {r: full_map[r] for r in allowed}


class Generator(nn.Module):
	def __init__(self, latent_dim: int = 512, max_resolution: int = 128):
		super().__init__()
		self.latent_dim = latent_dim
		self.max_resolution = max_resolution
		self.resolutions: List[int] = sorted(default_channel_map(max_resolution).keys())
		self.channels: Dict[int, int] = default_channel_map(max_resolution)

		# Initial 4x4 block
		c4 = self.channels[4]
		self.fc = EqualizedLRLinear(latent_dim, c4 * 4 * 4)
		self.initial_conv = nn.Sequential(
			nn.LeakyReLU(0.2),
			PixelNorm(),
			EqualizedLRConv2d(c4, c4, 3, padding=1),
			nn.LeakyReLU(0.2),
			PixelNorm(),
		)

		# Progressive generator blocks (each upsamples)
		blocks = {}
		to_rgbs = {}
		for idx, res in enumerate(self.resolutions):
			if res == 4:
				to_rgbs[str(res)] = EqualizedLRConv2d(c4, 3, 1, padding=0)
				continue
			prev_res = self.resolutions[idx - 1]
			in_ch = self.channels[prev_res]
			out_ch = self.channels[res]
			blocks[str(res)] = GeneratorConvBlock(in_ch, out_ch)
			to_rgbs[str(res)] = EqualizedLRConv2d(out_ch, 3, 1, padding=0)

		self.blocks = nn.ModuleDict(blocks)
		self.to_rgbs = nn.ModuleDict(to_rgbs)

	def forward(self, z: torch.Tensor, current_resolution: int, alpha: float = 1.0) -> torch.Tensor:
		assert current_resolution in self.channels, f"Unsupported resolution {current_resolution}"

		batch_size = z.shape[0]
		x = self.fc(z).view(batch_size, self.channels[4], 4, 4)
		x = self.initial_conv(x)

		if current_resolution == 4:
			return self.to_rgbs["4"](x)

		# Walk up the resolutions
		prev_features = x
		for res in self.resolutions:
			if res == 4:
				continue
			block = self.blocks[str(res)]
			if res == current_resolution:
				# Compute blended output for fade-in
				out_new = self.to_rgbs[str(res)](block(prev_features))
				out_prev = F.interpolate(self.to_rgbs[str(res // 2)](prev_features), scale_factor=2, mode="nearest")
				return alpha * out_new + (1.0 - alpha) * out_prev
			else:
				prev_features = block(prev_features)

		# Fallback (should not reach here if current_resolution is valid)
		return self.to_rgbs[str(self.resolutions[-1])](prev_features)