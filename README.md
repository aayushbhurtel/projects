# ProGAN (Progressive Growing GAN) for Faces

This project implements Progressive Growing of GANs (ProGAN) in PyTorch, aligned with the Paperspace tutorial.

## Setup

1. Create a virtual environment (recommended) and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare your dataset directory of face images (any nested folder structure is fine). The loader will recursively find common image files. If you're using CelebA, point to the root folder with images.

## Training

```bash
python train_progan.py \
  --data_root /path/to/images \
  --output_dir ./runs/progan \
  --max_images 10000 \
  --max_resolution 128 \
  --batch_size 32
```

- The training uses WGAN-GP loss.
- Progressive resolutions default to: 4, 8, 16, 32, 64, 128 (configurable up to 1024 if your GPU allows).
- At each resolution, training runs a fade-in phase followed by a stabilization phase.

## Sampling

Checkpoints and sample image grids are saved in `output_dir`. Samples are normalized to `[-1, 1]` during training and saved as PNGs.

## Notes

- Default configuration caps the dataset to 10,000 images to reduce compute requirement.
- For best results, use a face dataset (e.g., CelebA). Center-cropped faces generally produce better results.
