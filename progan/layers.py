import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNorm(nn.Module):
	def __init__(self, epsilon: float = 1e-8):
		super().__init__()
		self.epsilon = epsilon

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.epsilon)


class EqualizedLRLinear(nn.Module):
	def __init__(
		self,
		in_features: int,
		out_features: int,
		bias: bool = True,
		gain: float = math.sqrt(2.0),
	):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features

		# Weight scaling per Karras et al. (Equalized LR)
		self.weight = nn.Parameter(torch.randn(out_features, in_features))
		self.scale = gain / math.sqrt(in_features)
		self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		weight = self.weight * self.scale
		return F.linear(x, weight, self.bias)


class EqualizedLRConv2d(nn.Module):
	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		stride: int = 1,
		padding: int = 0,
		bias: bool = True,
		gain: float = math.sqrt(2.0),
	):
		super().__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		fan_in = in_channels * kernel_size * kernel_size
		self.weight = nn.Parameter(
			torch.randn(out_channels, in_channels, kernel_size, kernel_size)
		)
		self.scale = gain / math.sqrt(fan_in)
		self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		weight = self.weight * self.scale
		return F.conv2d(x, weight, self.bias, stride=self.stride, padding=self.padding)


class MinibatchStdDev(nn.Module):
	def __init__(self, group_size: int = 4, epsilon: float = 1e-8):
		super().__init__()
		self.group_size = group_size
		self.epsilon = epsilon

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, _, height, width = x.shape
		group = min(self.group_size, batch_size)
		# Ensure group divides batch_size to allow reshaping
		while batch_size % group != 0 and group > 1:
			group -= 1
		if group <= 1:
			std = torch.sqrt(x.var(dim=0, unbiased=False) + self.epsilon)
			mean_std = std.mean().view(1, 1, 1, 1)
			mean_std = mean_std.repeat(batch_size, 1, height, width)
			return torch.cat([x, mean_std], dim=1)

		G = group
		M = batch_size // G
		y = x.view(G, M, -1, height, width)
		y = y - y.mean(dim=1, keepdim=True)
		y = torch.sqrt(y.var(dim=1, unbiased=False) + self.epsilon)
		mean_std = y.mean(dim=[2, 3, 4], keepdim=True)
		mean_std = mean_std.view(batch_size, 1, 1, 1)
		mean_std = mean_std.repeat(1, 1, height, width)
		return torch.cat([x, mean_std], dim=1)


class GeneratorConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.conv1 = EqualizedLRConv2d(in_channels, out_channels, 3, padding=1)
		self.conv2 = EqualizedLRConv2d(out_channels, out_channels, 3, padding=1)
		self.pixel_norm = PixelNorm()
		self.activation = nn.LeakyReLU(0.2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# Upsample by factor 2 using nearest neighbor
		x = F.interpolate(x, scale_factor=2, mode="nearest")
		x = self.activation(self.conv1(x))
		x = self.pixel_norm(x)
		x = self.activation(self.conv2(x))
		x = self.pixel_norm(x)
		return x


class DiscriminatorConvBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int):
		super().__init__()
		self.conv1 = EqualizedLRConv2d(in_channels, in_channels, 3, padding=1)
		self.conv2 = EqualizedLRConv2d(in_channels, out_channels, 3, padding=1)
		self.activation = nn.LeakyReLU(0.2)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.activation(self.conv1(x))
		x = self.activation(self.conv2(x))
		x = F.avg_pool2d(x, kernel_size=2)
		return x