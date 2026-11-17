# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Tuple

import torch


def tile_spans(length: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    """Cover one axis on the latent grid with balanced, overlapping spans."""
    if length <= size:
        return [(0, length)]

    stride = size - overlap
    count = 1 + (length - size + stride - 1) // stride
    # Distribute the work evenly so the last tile isn't a narrow strip.
    width, extra = divmod(length + (count - 1) * overlap, count)
    spans = []
    start = 0
    for i in range(count):
        end = start + width + (i < extra)
        spans.append((start, end))
        start = end - overlap
    return spans


def _axis_weights(spans, overlap, scale, device):
    spans = [(start * scale, end * scale) for start, end in spans]
    overlap *= scale
    ramp = torch.arange(1, overlap + 1, device=device, dtype=torch.float32) / (overlap + 1)
    total = torch.zeros(spans[-1][1], device=device, dtype=torch.float32)
    weights = []
    for i, (start, end) in enumerate(spans):
        weight = torch.ones(end - start, device=device, dtype=torch.float32)
        if overlap and i > 0:
            weight[:overlap] *= ramp
        if overlap and i < len(spans) - 1:
            weight[-overlap:] *= ramp.flip(0)
        total[start:end] += weight
        weights.append(weight)
    # Normalise even when three or more spans overlap. Separable weights also
    # normalise corners where several rows and columns meet.
    return [weight / total[start:end] for (start, end), weight in zip(spans, weights)]


def spatial_tiled_forward(
    x: torch.Tensor,
    forward: Callable[[torch.Tensor], torch.Tensor],
    *,
    size: int,
    overlap: int,
    input_scale: int,
    output_scale: int,
) -> torch.Tensor:
    """Apply a BCTHW VAE one spatial tile at a time, blending on its output grid."""
    input_h, input_w = x.shape[-2:]
    height, width = input_h // input_scale, input_w // input_scale
    if min(height, width) < 1:
        raise ValueError(f"Both spatial dimensions must be at least {input_scale}.")
    rows = tile_spans(height, size, overlap)
    cols = tile_spans(width, size, overlap)
    weights_y = _axis_weights(rows, overlap, output_scale, x.device)
    weights_x = _axis_weights(cols, overlap, output_scale, x.device)

    output = None
    for (y0, y1), wy in zip(rows, weights_y):
        for (x0, x1), wx in zip(cols, weights_x):
            # Keep the original right/bottom pixels even if the encoder rounds
            # the latent dimensions down: these pixels still affect its edge.
            tile = forward(x[
                ...,
                y0 * input_scale : input_h if y1 == height else y1 * input_scale,
                x0 * input_scale : input_w if x1 == width else x1 * input_scale,
            ])
            if output is None:
                output = tile.new_zeros((*tile.shape[:-2], height * output_scale, width * output_scale))
            weight = wy[:, None] * wx[None, :]
            output[
                ...,
                y0 * output_scale : y1 * output_scale,
                x0 * output_scale : x1 * output_scale,
            ].addcmul_(tile, weight)
            del tile
    return output
