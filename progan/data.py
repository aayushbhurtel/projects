import os
from typing import List, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(root: str, max_images: Optional[int] = None) -> List[str]:
	paths: List[str] = []
	for base, _, files in os.walk(root):
		for name in files:
			ext = os.path.splitext(name)[1].lower()
			if ext in IMG_EXTS:
				paths.append(os.path.join(base, name))
				if max_images is not None and len(paths) >= max_images:
					return paths
	return paths


class ProgressiveImageDataset(Dataset):
	def __init__(
		self,
		root: str,
		resolution: int,
		max_images: Optional[int] = 10000,
	):
		self.paths = find_images(root, max_images=max_images)
		self.resolution = resolution
		self.transform = T.Compose([
			T.Resize((resolution, resolution), interpolation=InterpolationMode.BILINEAR),
			T.ToTensor(),
			T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
		])

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, idx: int) -> torch.Tensor:
		path = self.paths[idx]
		with Image.open(path) as img:
			img = img.convert("RGB")
			return self.transform(img)