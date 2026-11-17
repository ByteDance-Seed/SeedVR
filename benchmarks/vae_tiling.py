# SPDX-License-Identifier: Apache-2.0

"""Compare full-frame vs spatially tiled SeedVR VAE inference on one GPU."""

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter

import torch

from common.config import create_object, load_config
from models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d


ROOT = Path(__file__).resolve().parents[1]
MIB = 1024**2


def read_clip(path, frames, height, width):
    if path is None:
        return torch.rand(1, 3, frames, height, width) * 2 - 1

    import cv2

    capture = cv2.VideoCapture(str(path))
    rgb = []
    try:
        for _ in range(frames):
            ok, image = capture.read()
            if not ok:
                raise ValueError(f"Could not read {frames} frames from {path}")
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)
            rgb.append(torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    return torch.stack(rgb).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1


def measure(label, run, clear_cache, warmup, repeat):
    durations, peaks = [], []
    # Start each mode with the same allocator and cache conditions.
    clear_cache()
    torch.cuda.empty_cache()
    try:
        for i in range(warmup + repeat):
            clear_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            result = run()
            torch.cuda.synchronize()
            if i >= warmup:
                durations.append(perf_counter() - start)
                peaks.append(torch.cuda.max_memory_allocated() / MIB)
            if i == warmup + repeat - 1:
                output = result.cpu()
            del result
        stats = {
            "median_s": statistics.median(durations),
            "times_s": durations,
            "peak_allocated_mib": max(peaks),
        }
    except torch.cuda.OutOfMemoryError:
        output, stats = None, {"oom": True}
    finally:
        clear_cache()
    print(f"{label}: {stats}", flush=True)
    return output, stats


def error_metrics(reference, actual, pixels=False):
    if reference is None or actual is None:
        return None
    a, b = reference.float(), actual.float()
    if pixels:
        a, b = a.clamp(-1, 1), b.clamp(-1, 1)
    difference = b - a
    square = difference.square()
    mse = square.mean()
    metrics = {"mae": difference.abs().mean().item(), "rmse": mse.sqrt().item()}
    if pixels:
        worst_mse = square.mean(dim=(0, 1, 3, 4)).max()
        metrics["psnr_db"] = (10 * torch.log10(4 / mse)).item() if mse else None
        metrics["worst_frame_psnr_db"] = (10 * torch.log10(4 / worst_mse)).item() if worst_mse else None
        if difference.shape[2] > 1:
            # Error in the change from one frame to the next, in [0,1] pixel units.
            metrics["frame_delta_mae_01"] = difference.diff(dim=2).abs().mean().item() / 2
    return metrics


def save_preview(path, source, full, tiled, label):
    import cv2
    import numpy as np

    frame = source.shape[2] // 2

    def pixels(video):
        return video[0, :, frame].float().clamp(-1, 1).add(1).mul(127.5).round().byte().permute(1, 2, 0).numpy()

    source_rgb, full_rgb, tiled_rgb = map(pixels, (source, full, tiled))
    error = np.abs(tiled_rgb.astype(np.float32) - full_rgb).mean(axis=-1) / 255
    heat = cv2.applyColorMap(np.clip(error * 2550, 0, 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

    def panel(bgr, title):
        bgr = cv2.copyMakeBorder(bgr, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
        cv2.putText(bgr, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA)
        return bgr

    original, baseline, candidate = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in (source_rgb, full_rgb, tiled_rgb)]
    preview = np.vstack([
        np.hstack([panel(original, f"Input (frame {frame})"), panel(baseline, "Full-frame VAE")]),
        np.hstack([panel(candidate, f"Tiled VAE ({label})"), panel(heat, "Mean RGB absolute error (0 to 0.10)")]),
    ])
    if not cv2.imwrite(str(path), preview):
        raise OSError(f"Could not save {path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--checkpoint", type=Path, default=ROOT / "ckpts/ema_vae.pth")
    weights.add_argument("--random-weights", action="store_true", help="Smoke test only; quality metrics are not meaningful.")
    parser.add_argument("--video", type=Path, help="Optional input clip; otherwise use seeded random pixels.")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--tile", type=int, nargs=2, action="append", metavar=("SIZE", "OVERLAP"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.frames < 1 or (args.frames - 1) % 4 or min(args.height, args.width) < 8 or args.height % 8 or args.width % 8:
        parser.error("Frames must be 1 + 4k; height and width must be positive multiples of 8.")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("Warmup must be nonnegative and repeat must be positive.")
    if not torch.cuda.is_available():
        parser.error("The benchmark requires a CUDA GPU.")

    torch.manual_seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.backends.cudnn.benchmark = False
    dtype = getattr(torch, args.dtype)
    config = load_config(str(ROOT / "configs_3b/main.yaml")).vae
    vae = create_object(config.model).eval().requires_grad_(False)
    if not args.random_weights:
        if args.checkpoint.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(args.checkpoint))
        else:
            state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
        vae.load_state_dict(state, strict=True)
        del state
    vae.to(device="cuda", dtype=dtype)
    vae.set_causal_slicing(**config.slicing)
    vae.set_memory_limit(**config.memory_limit)
    convolutions = [m for m in vae.modules() if isinstance(m, InflatedCausalConv3d)]

    def clear_cache():
        for module in convolutions:
            module.memory = None

    def decode(latent):
        video = vae.decode(latent).sample
        return video.unsqueeze(2) if video.ndim == 4 else video

    def set_tiling(tile):
        if tile is None:
            vae.disable_tiling()
        else:
            vae.enable_tiling(tile_size=tile[0], tile_overlap=tile[1])

    x = read_clip(args.video, args.frames, args.height, args.width).to(device="cuda", dtype=dtype)
    modes = {"untiled": None, **{f"tile_{s}_{o}": (s, o) for s, o in args.tile or [(512, 128)]}}
    report = {
        "gpu": torch.cuda.get_device_name(),
        "torch": str(torch.__version__),
        "dtype": args.dtype,
        "input_shape_bcthw": list(x.shape),
        "video": str(args.video) if args.video else "random pixels (seed 42)",
        "checkpoint": "random weights (seed 42)" if args.random_weights else str(args.checkpoint),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "modes": {},
    }
    print({k: v for k, v in report.items() if k != "modes"}, flush=True)
    outputs = {}
    for label, tile in modes.items():
        set_tiling(tile)
        outputs[label], stats = measure(
            f"{label} encode", lambda: vae.encode(x).posterior.mode(), clear_cache, args.warmup, args.repeat
        )
        report["modes"][label] = {"tile_size_overlap": tile, "encode": stats}

    shared = next((label for label, z in outputs.items() if z is not None), None)
    report["shared_decode_latent"] = shared
    if shared is not None:
        z = outputs[shared].to("cuda")
        full_video = None
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        for label, tile in modes.items():
            set_tiling(tile)
            video, stats = measure(f"{label} decode", lambda: decode(z), clear_cache, args.warmup, args.repeat)
            metrics = report["modes"][label]
            metrics["decode_same_latent"] = stats
            if tile is None:
                full_video = video
                if shared == "untiled":
                    metrics["reconstruction_vs_input"] = error_metrics(x.cpu(), full_video, pixels=True)
            else:
                metrics["latent_vs_untiled"] = error_metrics(outputs["untiled"], outputs[label])
                metrics["decode_vs_untiled"] = error_metrics(full_video, video, pixels=True)
                if video is not None and outputs[label] is not None:
                    # Untimed round trip with the tiled encoder's own latents.
                    reconstruction = decode(outputs[label].to("cuda")).cpu()
                    source = x.cpu()
                    metrics["reconstruction_vs_input"] = error_metrics(source, reconstruction, pixels=True)
                    if shared == "untiled":
                        metrics["reconstruction_vs_untiled"] = error_metrics(full_video, reconstruction, pixels=True)
                        if args.output_dir and full_video is not None:
                            save_preview(args.output_dir / f"{label}.png", source, full_video, reconstruction, label)

    result = json.dumps(report, indent=2)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
