import argparse
import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from progan.generator import Generator
from progan.discriminator import Discriminator
from progan.data import ProgressiveImageDataset
from progan.utils import (
	requires_grad,
	sample_latents,
	save_image_grid,
	compute_gradient_penalty,
)


def get_device() -> torch.device:
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_size_for_resolution(base_batch: int, resolution: int) -> int:
	# Conservative defaults; adjust to your GPU memory
	max_per_res = {
		4: 128,
		8: 128,
		16: 64,
		32: 32,
		64: 16,
		128: 8,
		256: 4,
		512: 2,
		1024: 1,
	}
	return min(base_batch, max_per_res.get(resolution, base_batch))


def make_dataloader(data_root: str, resolution: int, max_images: int, batch_size: int, num_workers: int) -> DataLoader:
	dataset = ProgressiveImageDataset(root=data_root, resolution=resolution, max_images=max_images)
	return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)


def train(args: argparse.Namespace) -> None:
	device = get_device()
	os.makedirs(args.output_dir, exist_ok=True)

	gen = Generator(latent_dim=args.latent_dim, max_resolution=args.max_resolution).to(device)
	disc = Discriminator(max_resolution=args.max_resolution).to(device)

	optim_g = optim.Adam(gen.parameters(), lr=args.lr, betas=(0.0, 0.99))
	optim_d = optim.Adam(disc.parameters(), lr=args.lr, betas=(0.0, 0.99))

	resolutions = sorted(set(gen.resolutions))

	global_step = 0
	for resolution in resolutions:
		batch_size = batch_size_for_resolution(args.batch_size, resolution)
		loader = make_dataloader(args.data_root, resolution, args.max_images, batch_size, args.num_workers)
		if len(loader) == 0:
			print(f"No images found for resolution {resolution}.")
			continue

		print(f"Training at resolution {resolution}x{resolution} with batch_size={batch_size}")

		# Fade-in phase
		if resolution == 4:
			# 4x4 has no fade-in blending; treat as stabilize
			epochs_fade = 0
		else:
			epochs_fade = args.epochs_fade

		total_fade_steps = epochs_fade * len(loader) if epochs_fade > 0 else 0
		fade_step = 0

		for phase, num_epochs in [("fade", epochs_fade), ("stabilize", args.epochs_stable)]:
			if num_epochs == 0:
				continue

			progress = tqdm(range(num_epochs), desc=f"{phase} {resolution}")
			for _ in progress:
				for real_images in loader:
					real_images = real_images.to(device, non_blocking=True)

					# Determine alpha
					if phase == "fade" and total_fade_steps > 0:
						alpha = min(1.0, (fade_step + 1) / total_fade_steps)
						fade_step += 1
					else:
						alpha = 1.0

					# Train Discriminator
					requires_grad(gen, False)
					requires_grad(disc, True)
					optim_d.zero_grad(set_to_none=True)

					z = sample_latents(real_images.size(0), args.latent_dim, device)
					with torch.no_grad():
						fake_images = gen(z, current_resolution=resolution, alpha=alpha)

					real_logits = disc(real_images, current_resolution=resolution, alpha=alpha).view(-1)
					fake_logits = disc(fake_images, current_resolution=resolution, alpha=alpha).view(-1)

					d_loss = fake_logits.mean() - real_logits.mean()
					gp = compute_gradient_penalty(disc, real_images, fake_images, resolution, alpha, device, lambda_gp=args.lambda_gp)
					d_loss_total = d_loss + gp
					d_loss_total.backward()
					optim_d.step()

					# Train Generator
					requires_grad(gen, True)
					requires_grad(disc, False)
					optim_g.zero_grad(set_to_none=True)
					z = sample_latents(real_images.size(0), args.latent_dim, device)
					fake_images = gen(z, current_resolution=resolution, alpha=alpha)
					fake_logits = disc(fake_images, current_resolution=resolution, alpha=alpha).view(-1)
					g_loss = -fake_logits.mean()
					g_loss.backward()
					optim_g.step()

					global_step += 1

					if global_step % args.sample_every == 0:
						with torch.inference_mode():
							z = sample_latents(args.sample_grid ** 2, args.latent_dim, device)
							samples = gen(z, current_resolution=resolution, alpha=alpha)
							save_path = os.path.join(args.output_dir, f"samples_res{resolution}_step{global_step}.png")
							save_image_grid(samples, save_path, nrow=args.sample_grid)

					if global_step % args.checkpoint_every == 0:
						checkpoint = {
							"gen": gen.state_dict(),
							"disc": disc.state_dict(),
							"optim_g": optim_g.state_dict(),
							"optim_d": optim_d.state_dict(),
							"resolution": resolution,
							"alpha": alpha,
							"global_step": global_step,
						}
						torch.save(checkpoint, os.path.join(args.output_dir, f"checkpoint_step{global_step}.pt"))

			# End of phase; save a resolution milestone checkpoint
			ckpt_name = f"checkpoint_res{resolution}_{phase}.pt"
			checkpoint = {
				"gen": gen.state_dict(),
				"disc": disc.state_dict(),
				"optim_g": optim_g.state_dict(),
				"optim_d": optim_d.state_dict(),
				"resolution": resolution,
				"alpha": 1.0,
				"global_step": global_step,
			}
			torch.save(checkpoint, os.path.join(args.output_dir, ckpt_name))

	print("Training completed.")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train ProGAN with progressive growing and WGAN-GP")
	parser.add_argument("--data_root", type=str, required=True, help="Root directory of images")
	parser.add_argument("--output_dir", type=str, default="./runs/progan", help="Directory to save outputs")
	parser.add_argument("--max_images", type=int, default=10000, help="Max images to load due to compute limits")
	parser.add_argument("--max_resolution", type=int, default=128, help="Maximum resolution to train up to")
	parser.add_argument("--latent_dim", type=int, default=512)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--num_workers", type=int, default=4)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--lambda_gp", type=float, default=10.0)
	parser.add_argument("--epochs_fade", type=int, default=5)
	parser.add_argument("--epochs_stable", type=int, default=5)
	parser.add_argument("--sample_every", type=int, default=500)
	parser.add_argument("--checkpoint_every", type=int, default=2000)
	parser.add_argument("--sample_grid", type=int, default=6, help="Grid size for sampling (grid=G => GxG images)")
	args = parser.parse_args()

	torch.backends.cudnn.benchmark = True
	train(args)