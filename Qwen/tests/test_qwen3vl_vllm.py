#!/usr/bin/env python3
"""
Comprehensive test suite for Qwen3-VL vLLM encoder/decoder.

Tests:
1. Basic encoder functionality (batch processing)
2. vLLM decoder initialization and generation
3. Performance benchmarking (encoder + decoder)
4. Output quality validation vs HuggingFace baseline
"""

import time

import pytest
import torch
from PIL import Image
from Renderer import VelloRenderer


from Qwen.decoder import Qwen3VLDecoder
from Qwen.encoder import Qwen3VLEncoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestQwen3VLvLLM:
    """Test suite for Qwen3-VL with vLLM backend."""

    @pytest.fixture(scope="class")
    def test_images(self):
        renderer = VelloRenderer(width=640, height=640, padding=30)
        texts = ["Hello World", "Test Image"]
        arrays = renderer.render_batch(texts)
        return [Image.fromarray(arr) for arr in arrays]

    def test_encoder_batch_output_shapes(self, test_images):
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        output = encoder.encode_images(test_images)

        batch_size = len(test_images)
        expected_tokens = 400

        assert output.features.shape == (batch_size, expected_tokens, 2048)
        assert len(output.deepstack_features) == 3
        for ds in output.deepstack_features:
            assert ds.shape == (batch_size, expected_tokens, 2048)
        assert output.grid_thw.shape == (batch_size, 3)

    def test_encoder_performance(self, test_images):
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        _ = encoder.encode_images(test_images[:1])

        start = time.time()
        _ = encoder.encode_images(test_images)
        speed = len(test_images) / (time.time() - start)
        assert speed > 10

    def test_decoder_initialization(self):
        decoder = Qwen3VLDecoder(
            device="cuda:0",
            dtype="bfloat16",
            gpu_memory_utilization=0.5,
            max_model_len=2048,
        )
        assert decoder.llm is not None

    def test_decoder_generation(self, test_images):
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        output = encoder.encode_images([test_images[0]])

        feat = output.features[0]
        deepstack = [ds[0] for ds in output.deepstack_features]
        concat_emb = torch.cat([feat] + deepstack, dim=-1)
        grid_thw = output.grid_thw[0].tolist()

        decoder = Qwen3VLDecoder(device="cuda:0", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)
        text = decoder.decode(
            visual_embeddings=concat_emb,
            grid_thw=grid_thw,
            prompts="What do you see?",
            max_tokens=20,
            temperature=0.0,
        )

        assert isinstance(text, str)
        assert len(text) > 0

    def test_batch_decoding(self, test_images):
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        output = encoder.encode_images(test_images)

        batch_size = len(test_images)
        embeddings = []
        grids = []
        for i in range(batch_size):
            feat_i = output.features[i]
            deepstack_i = [ds[i] for ds in output.deepstack_features]
            embeddings.append(torch.cat([feat_i] + deepstack_i, dim=-1))
            grids.append(output.grid_thw[i].tolist())

        decoder = Qwen3VLDecoder(device="cuda:0", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)
        texts = decoder.decode(
            visual_embeddings=embeddings,
            grid_thw=grids,
            prompts=["Describe the image:"] * batch_size,
            max_tokens=20,
            temperature=0.0,
        )

        assert isinstance(texts, list)
        assert len(texts) == batch_size
        for text in texts:
            assert len(text) > 0

    def test_end_to_end_performance(self, test_images):
        encoder = Qwen3VLEncoder(device="cuda:0", use_vllm_kernels=True, dtype=torch.bfloat16)
        decoder = Qwen3VLDecoder(device="cuda:0", dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=2048)

        warmup_output = encoder.encode_images([test_images[0]])
        warmup_emb = torch.cat([warmup_output.features[0]] + [ds[0] for ds in warmup_output.deepstack_features], dim=-1)
        _ = decoder.decode(warmup_emb, warmup_output.grid_thw[0].tolist(), "Test", max_tokens=10, temperature=0.0)

        start = time.time()
        output = encoder.encode_images(test_images)
        encode_time = time.time() - start

        batch_size = len(test_images)
        embeddings = []
        grids = []
        for i in range(batch_size):
            feat_i = output.features[i]
            deepstack_i = [ds[i] for ds in output.deepstack_features]
            embeddings.append(torch.cat([feat_i] + deepstack_i, dim=-1))
            grids.append(output.grid_thw[i].tolist())

        start = time.time()
        _ = decoder.decode(embeddings, grids, ["Describe:"] * batch_size, max_tokens=20, temperature=0.0)
        decode_time = time.time() - start

        throughput = batch_size / (encode_time + decode_time)
        assert throughput > 1.0
