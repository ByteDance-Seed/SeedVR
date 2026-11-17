# SPDX-License-Identifier: Apache-2.0

import unittest

import torch
import torch.nn.functional as F

from models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
from models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d
from models.video_vae_v3.modules.spatial_tiling import spatial_tiled_forward


class VideoVAETilingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, threads)
        cls.vae = VideoAutoencoderKLWrapper(
            block_out_channels=(8, 8, 8, 8),
            down_block_types=("DownEncoderBlock3D",) * 4,
            up_block_types=("UpDecoderBlock3D",) * 4,
            norm_num_groups=4,
            freeze_encoder=False,
            spatial_downsample_factor=8,
            temporal_downsample_factor=4,
        ).eval()

    def setUp(self):
        self.vae.disable_tiling()
        self.vae.disable_slicing()

    def test_overlap_add_keeps_alignment_and_scale(self):
        video = torch.rand(2, 3, 3, 41, 57)

        def encode(tile):
            return F.avg_pool3d(tile, kernel_size=(1, 4, 4))

        def decode(tile):
            return tile.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)

        # With this overlap, more than two tiles can contribute along either axis.
        latent = spatial_tiled_forward(video, encode, size=5, overlap=4, input_scale=4, output_scale=1)
        torch.testing.assert_close(latent, encode(video))
        actual = spatial_tiled_forward(latent, decode, size=5, overlap=4, input_scale=1, output_scale=4)
        torch.testing.assert_close(actual, decode(encode(video)))

    @torch.inference_mode()
    def test_small_image_falls_back_to_full_frame(self):
        x = torch.randn(1, 3, 24, 31)
        z = self.vae.encode(x).posterior.mode()
        expected = self.vae.decode(z).sample
        self.vae.enable_tiling(tile_size=32, tile_overlap=8)
        torch.testing.assert_close(self.vae.encode(x).posterior.mode(), z, rtol=0, atol=0)
        torch.testing.assert_close(self.vae.decode(z).sample, expected, rtol=0, atol=0)
        self.assertEqual(expected.shape, (1, 3, 24, 24))

    @torch.inference_mode()
    def test_single_frame_can_tile_only_width(self):
        self.vae.enable_tiling(tile_size=32, tile_overlap=8)
        encoded = self.vae.encode(torch.randn(2, 3, 24, 53))
        self.assertEqual(encoded.latent.shape, (2, 4, 3, 6))
        self.assertEqual(self.vae.decode(encoded.latent).sample.shape, (2, 3, 24, 48))

    @torch.inference_mode()
    def test_tiling_reuses_causal_history_only_within_each_tile(self):
        x = torch.randn(1, 3, 9, 45, 53)
        full_z = self.vae.encode(x).posterior.mode()
        full_video = self.vae.decode(full_z).sample

        self.vae.enable_tiling(tile_size=32, tile_overlap=16)
        z = self.vae.encode(x).posterior.mode()
        video = self.vae.decode(z).sample
        self.assertEqual(z.shape, full_z.shape)
        self.assertEqual(video.shape, full_video.shape)
        self.assertFalse(torch.equal(video, full_video))

        self.vae.set_causal_slicing(split_size=4, memory_device="same")
        self.vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=None)
        self.addCleanup(self.vae.set_memory_limit, None, None)
        torch.testing.assert_close(self.vae.encode(x).posterior.mode(), z, rtol=1e-4, atol=2e-5)
        torch.testing.assert_close(self.vae.decode(z).sample, video, rtol=1e-4, atol=2e-5)
        for module in self.vae.modules():
            if isinstance(module, InflatedCausalConv3d):
                self.assertIsNone(module.memory)

        self.vae.disable_tiling()
        torch.testing.assert_close(self.vae.encode(x).posterior.mode(), full_z, rtol=1e-4, atol=2e-5)

    def test_tile_settings_must_align_to_spatial_factor(self):
        for size, overlap in [(31, 8), (32, 9), (32, 32), (32, -8), (0, 0)]:
            with self.subTest(size=size, overlap=overlap), self.assertRaises(ValueError):
                self.vae.enable_tiling(tile_size=size, tile_overlap=overlap)
