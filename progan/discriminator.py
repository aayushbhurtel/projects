from typing import Dict, List

import torch
import torch.nn as nn

from .layers import EqualizedLRConv2d, DiscriminatorConvBlock, MinibatchStdDev


def default_channel_map(max_resolution: int) -> Dict[int, int]:
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


class Discriminator(nn.Module):
	def __init__(self, max_resolution: int = 128):
		super().__init__()
		self.max_resolution = max_resolution
		self.resolutions: List[int] = sorted(default_channel_map(max_resolution).keys())
		self.channels: Dict[int, int] = default_channel_map(max_resolution)

		from_rgbs = {}
		blocks = {}
		for idx, res in enumerate(self.resolutions):
			c = self.channels[res]
			from_rgbs[str(res)] = EqualizedLRConv2d(3, c, 1)
			if res == 4:
				continue
			prev_res = self.resolutions[idx - 1]
			in_ch = self.channels[res]
			out_ch = self.channels[prev_res]
			blocks[str(res)] = DiscriminatorConvBlock(in_ch, out_ch)

		self.from_rgbs = nn.ModuleDict(from_rgbs)
		self.blocks = nn.ModuleDict(blocks)

		self.mbstd = MinibatchStdDev()
		c4 = self.channels[4]
		self.final_conv = EqualizedLRConv2d(c4 + 1, c4, 3, padding=1)
		self.final_dense = nn.Sequential(
			EqualizedLRConv2d(c4, c4, 4, padding=0),  # 4x4 -> 1x1
			nn.LeakyReLU(0.2),
			EqualizedLRConv2d(c4, 1, 1),
		)

		self.activation = nn.LeakyReLU(0.2)

	def forward(self, x: torch.Tensor, current_resolution: int, alpha: float = 1.0) -> torch.Tensor:
		assert current_resolution in self.channels, f"Unsupported resolution {current_resolution}"

		if current_resolution == 4:
			x = self.activation(self.from_rgbs["4"](x))
			x = self.mbstd(x)
			x = self.activation(self.final_conv(x))
			return self.final_dense(x).view(-1, 1)

		# Top-level blend between current and previous resolution
		x_new = self.activation(self.from_rgbs[str(current_resolution)](x))
		x_new = self.blocks[str(current_resolution)](x_new)

		x_prev = nn.functional.avg_pool2d(x, kernel_size=2)
		x_prev = self.activation(self.from_rgbs[str(current_resolution // 2)](x_prev))

		x = alpha * x_new + (1.0 - alpha) * x_prev

		# Cascade down to 4x4
		res = current_resolution // 2
		while res > 4:
			x = self.blocks[str(res)](x)
			res //= 2

		x = self.mbstd(x)
		x = self.activation(self.final_conv(x))
		logits = self.final_dense(x).view(-1, 1)
		return logits