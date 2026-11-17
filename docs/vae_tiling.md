# Spatial tiling in the SeedVR video VAE

## Upstream implementations

- [Wan2.2 TI2V-5B VAE](https://github.com/Wan-Video/Wan2.2/blob/main/wan/modules/vae2_2.py) reads the first frame on its own, then reads four frames at a time. The causal 3D convolutions carry short feature histories forward. The reference file does temporal chunking; [Diffusers' Wan implementation](https://github.com/huggingface/diffusers/blob/v0.36.0/src/diffusers/models/autoencoders/autoencoder_kl_wan.py#L1152-L1280) adds overlapping spatial tiles, clears the cache before each tile, and converts tile coordinates using the spatial compression ratio and the Wan2.2 patch size.
- [Diffusers' CogVideoX implementation](https://github.com/huggingface/diffusers/blob/v0.35.2/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L1166-L1298) starts each spatial tile with an empty `conv_cache` and reuses it across that tile's temporal chunks (eight sample frames or two latent frames per chunk, with the remainder in the first chunk). Both implementations blend neighboring outputs and discard duplicate overlap.

## SeedVR implementation

SeedVR's VAE compresses space by 8 and time by 4. Tile boundaries are aligned to latent cells, so the encoder and decoder agree on positions even when a frame is rectangular or its size is not a multiple of the configured tile size. Each tile uses the existing causal slicing path, including first-frame handling, and releases its cache before the next tile. Stitching uses normalized linear weights along both spatial axes and writes directly to an output canvas; it does not keep a list of completed tiles on the GPU.

On a single GPU with the default slicing configuration, a 17-frame sample is encoded as `5 + 4 + 4 + 4` frames. Its five latent frames are decoded as `2 + 1 + 1 + 1`. The first chunk initializes the convolution history and handles the special first frame; later chunks reuse that history without reinitializing it.

Enable it in any SeedVR or SeedVR2 inference script with `--vae_tile 512 128`. Both arguments are measured in sample pixels. The equivalent Python API is `vae.enable_tiling(tile_size=512, tile_overlap=128)` on the evaluation model inside `torch.inference_mode()`. `vae.disable_tiling()` restores whole-frame VAE inference. The existing temporal slicing remains independently configurable.

Tiling changes the spatial context available to GroupNorm and attention. Blending smooths transitions but cannot recover the full-frame context. Smaller tiles usually lower the activation peak; larger tiles generally retain more context. More overlap also increases the amount of repeated computation.

## Reproduce the GPU benchmark

From the repository root, with a CUDA build of PyTorch and the VAE dependencies installed, run:

```bash
python -m benchmarks.vae_tiling \
  --checkpoint ckpts/ema_vae.pth --video sample.mp4 \
  --frames 17 --height 720 --width 1248 \
  --tile 768 128 --tile 512 128 --tile 384 128 --output-dir results/vae_tiling
```

For an isolated benchmark environment with PyTorch already installed, the remaining dependencies are `diffusers==0.29.1`, `huggingface_hub==0.25.2`, `omegaconf==2.3.0`, `einops==0.7.0`, and `opencv-python-headless`.

The script accepts the [official VAE checkpoint](https://huggingface.co/ByteDance-Seed/SeedVR2-3B/blob/37255ff8cccfb01071b87f635a5948ca8d53117c/ema_vae.pth) or the [Comfy-Org FP16 conversion](https://huggingface.co/Comfy-Org/SeedVR2/blob/673340c8a66db62b84e4099def7d01d337ae12dc/vae/ema_vae_fp16.safetensors). It uses deterministic posterior means for the comparison, times encode and decode separately, and decodes the same latent in all modes to isolate decoder differences. It also compares each mode's complete VAE reconstruction against both the original frames and the untiled reconstruction. Reported peak CUDA allocations include the model, inputs, and outputs, and exclude unused allocator reservations. Timings are medians after warmup, synchronized on the GPU. The output directory contains JSON and a middle-frame comparison with an error heatmap. The benchmark measures only the VAE and requires no DiT checkpoint, Apex, or FlashAttention installation.

## Colab A100 results

Measured on one NVIDIA A100-SXM4 (40 GiB), with PyTorch `2.14.1+cu130`, Diffusers `0.29.1`, and BF16 inference using the Comfy-Org checkpoint above. The input was the first 17 frames of the [SeedVR demo clip `23_1_lq.mp4`](https://huggingface.co/datasets/Iceclear/SeedVR_VideoDemos/blob/main/seedvr_videos_crf23/aigc1k/23_1_lq.mp4) at its native 720×1248 resolution. Each mode used the default temporal slicing and memory limits, one warmup, and three timed runs.

| Tile / overlap (px) | Encode time (s) | Encode peak (GiB) | Decode time (s) | Decode peak (GiB) | Decode PSNR vs. untiled (dB) | Reconstruction PSNR vs. untiled (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Untiled | 1.66 | 9.70 | 3.75 | 15.59 | — | — |
| 768 / 128 | 1.59 | 6.17 | 3.72 | 9.99 | 42.96 | 41.13 |
| 512 / 128 | 1.91 | 3.10 | 4.42 | 5.17 | 40.56 | 38.07 |
| 384 / 128 | 2.66 | 1.94 | 5.89 | 3.21 | 39.55 | 36.05 |

On this clip, 512 / 128 cut peak allocations by 68% for encoding and 67% for decoding, at a time cost of 15% and 18%, respectively. The 768 / 128 setting kept more of the original spatial context and ran in about the same time as the untiled VAE. The PSNR columns measure agreement with the untiled result, which is a baseline for comparison rather than ground truth.
