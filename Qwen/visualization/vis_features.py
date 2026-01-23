#!/usr/bin/env python3
"""
Feature extraction and management module.

Provides unified interface for extracting features from different encoders:
- DPSK OCR (with SAM, CLIP components)
- Qwen3VL (with deepstack features)
- Qwen25VL
"""

import sys
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple
from dataclasses import dataclass
import torch
from PIL import Image
import numpy as np

# Add Qwen directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vis_core import (
    EncoderManager, EncoderConfig, EncoderType,
    FeatureSpec, ImagePreprocessor,
)


@dataclass
class FeatureData:
    """Container for extracted features."""
    features: np.ndarray  # [num_tokens, hidden_dim] or [pooled_dim]
    metadata: Dict[str, any]  # Additional info (shape, layer, etc.)
    image_path: Optional[Path] = None


class FeatureExtractor:
    """Unified feature extraction interface."""

    def __init__(
        self,
        encoder_configs: Optional[List[EncoderConfig]] = None,
    ):
        """
        Args:
            encoder_configs: List of encoder configurations (uses defaults if None)
        """
        if encoder_configs is None:
            # Use default configurations
            encoder_configs = [
                EncoderConfig(
                    encoder_type=EncoderType.DPSK,
                    model_path="deepseek-ai/DeepSeek-OCR",
                ),
                EncoderConfig(
                    encoder_type=EncoderType.QWEN3VL,
                    model_path="/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Instruct",
                ),
            ]

        self.encoder_configs = {cfg.encoder_type: cfg for cfg in encoder_configs}
        self.encoders = {}

    def get_encoder(self, encoder_type: EncoderType):
        """Get or create encoder."""
        if encoder_type not in self.encoders:
            if encoder_type not in self.encoder_configs:
                raise ValueError(f"No configuration for encoder type: {encoder_type}")
            self.encoders[encoder_type] = EncoderManager.get_encoder(
                self.encoder_configs[encoder_type]
            )
        return self.encoders[encoder_type]

    def extract(
        self,
        image: Union[Path, Image.Image],
        encoder_type: EncoderType,
        spec: FeatureSpec,
    ) -> FeatureData:
        """
        Extract features from image.

        Args:
            image: Image path or PIL Image
            encoder_type: Which encoder to use
            spec: Feature specification

        Returns:
            FeatureData with extracted features
        """
        # Load image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            pil_image = ImagePreprocessor.load_image(image_path)
        else:
            image_path = None
            pil_image = image

        # Extract features
        encoder = self.get_encoder(encoder_type)
        features = self._extract_features(encoder, pil_image, encoder_type, spec)

        # Create feature data
        feature_data = FeatureData(
            features=features,
            metadata={
                'encoder_type': encoder_type.value,
                'spec_name': spec.name,
                'layer': spec.layer,
                'feature_type': spec.feature_type,
                'pooling': spec.pooling,
                'shape': features.shape,
            },
            image_path=image_path,
        )

        return feature_data

    def _extract_features(
        self,
        encoder,
        image: Image.Image,
        encoder_type: EncoderType,
        spec: FeatureSpec,
    ) -> np.ndarray:
        """Extract features based on encoder type and spec."""
        if encoder_type == EncoderType.DPSK:
            return self._extract_dpsk_features(encoder, image, spec)
        elif encoder_type in (EncoderType.QWEN3VL, EncoderType.QWEN25VL):
            return self._extract_qwen_features(encoder, image, spec)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def _extract_dpsk_features(
        self,
        encoder,
        image: Image.Image,
        spec: FeatureSpec,
    ) -> np.ndarray:
        """Extract features from DPSK OCR encoder."""
        if spec.feature_type == 'sam_raw':
            # Raw SAM features before neck compression
            return self._extract_sam_raw(encoder, image, spec)
        else:
            # Use standard encode_images API
            return self._extract_dpsk_standard(encoder, image, spec)

    def _extract_sam_raw(
        self,
        encoder,
        image: Image.Image,
        spec: FeatureSpec,
    ) -> np.ndarray:
        """Extract raw SAM 40x40 features before neck."""
        # Preprocess image
        img_tensor = ImagePreprocessor.preprocess_for_encoder(image, EncoderType.DPSK)
        img_tensor = img_tensor.to(torch.bfloat16).to("cuda:0")

        with torch.no_grad():
            # Get SAM output after transformer blocks, before neck
            x = encoder.sam_model.patch_embed(img_tensor)  # [1, 768, 40, 40]
            for block in encoder.sam_model.blocks:
                x = block(x)
            # x shape: [1, 40, 40, 768] or [1, 768, 40, 40]

        # Convert to numpy
        features = x.float().cpu().numpy()

        # Pool if requested
        if spec.pooling == 'mean':
            # Global average pooling
            features = features.mean(axis=(1, 2))  # [768]
        elif spec.pooling == 'none':
            # Keep spatial structure
            pass

        return features

    def _extract_dpsk_standard(
        self,
        encoder,
        image: Image.Image,
        spec: FeatureSpec,
    ) -> np.ndarray:
        """Extract features using DPSK standard API."""
        return_intermediate = spec.layer != 'final'

        outputs = encoder.encode_images([image], return_intermediate=return_intermediate)

        if not outputs:
            raise RuntimeError("Failed to encode image")

        # The output is directly a tensor when return_intermediate=False
        # Or an EncoderOutput object when return_intermediate=True
        if return_intermediate:
            output = outputs[0]  # EncoderOutput object

            # Intermediate features
            if not output.intermediate_features:
                raise ValueError(f"No intermediate features available")

            # Parse layer index (e.g., "intermediate_0" -> index 0)
            if spec.layer.startswith('intermediate_'):
                idx = int(spec.layer.split('_')[1])
                features = output.intermediate_features[idx]
            else:
                raise ValueError(f"Invalid layer spec: {spec.layer}")

            # Split CLS and visual tokens
            features = features.float().cpu().numpy()  # [num_tokens, hidden_dim]

            if spec.feature_type == 'cls':
                features = features[0]  # CLS token
            elif spec.feature_type == 'visual':
                features = features[1:]  # Remove CLS token
                if spec.pooling == 'mean':
                    features = features.mean(axis=0)
            else:
                raise ValueError(f"Unknown feature_type: {spec.feature_type}")
        else:
            # Final output - directly a tensor
            features = outputs[0]  # Tensor [num_tokens, hidden_dim]

            if spec.feature_type == 'visual':
                features = features.float().cpu().numpy()
                if spec.pooling == 'mean':
                    features = features.mean(axis=0)  # Pool over tokens
            elif spec.feature_type == 'cls':
                # For DPSK final output, we don't have CLS token in the embeddings
                # Use the CLS token from the clip model if needed
                raise ValueError("CLS token not available in final DPSK output")
            else:
                raise ValueError(f"Unknown feature_type: {spec.feature_type}")

        return features

    def _extract_qwen_features(
        self,
        encoder,
        image: Image.Image,
        spec: FeatureSpec,
    ) -> np.ndarray:
        """Extract features from Qwen encoder."""
        output = encoder.encode_images([image])

        if spec.layer == 'final':
            # Final output features
            features = output.features[0].float().cpu().numpy()
        elif spec.layer.startswith('deepstack_'):
            # Deepstack intermediate features
            idx = int(spec.layer.split('_')[1])
            if idx >= len(output.deepstack_features):
                raise ValueError(f"Deepstack index {idx} out of range")
            features = output.deepstack_features[idx][0].float().cpu().numpy()
        else:
            raise ValueError(f"Invalid layer spec: {spec.layer}")

        # Pool if requested
        if spec.pooling == 'mean':
            features = features.mean(axis=0)  # Pool over tokens

        return features

    def extract_multiple(
        self,
        image: Union[Path, Image.Image],
        encoder_types: List[EncoderType],
        specs: List[FeatureSpec],
    ) -> Dict[Tuple[EncoderType, str], FeatureData]:
        """
        Extract multiple features from image.

        Returns:
            Dict mapping (encoder_type, spec_name) to FeatureData
        """
        results = {}
        for encoder_type in encoder_types:
            for spec in specs:
                key = (encoder_type, spec.name)
                try:
                    results[key] = self.extract(image, encoder_type, spec)
                except Exception as e:
                    print(f"Warning: Failed to extract {key}: {e}")
        return results

    def extract_batch(
        self,
        image_paths: List[Path],
        encoder_type: EncoderType,
        spec: FeatureSpec,
    ) -> List[FeatureData]:
        """Extract features from multiple images."""
        results = []
        for path in image_paths:
            try:
                results.append(self.extract(path, encoder_type, spec))
            except Exception as e:
                print(f"Warning: Failed to extract from {path}: {e}")
        return results

    def cleanup(self):
        """Clean up resources."""
        self.encoders.clear()
        EncoderManager.cleanup()


# Convenience functions for common extraction patterns

def extract_sam_comparison(image: Union[Path, Image.Image]) -> Dict[str, FeatureData]:
    """Extract SAM features at multiple compression stages.

    Returns:
        Dict mapping spec_name to FeatureData (string keys for convenience)

    Note:
        dpsk_final uses pooling='none' to preserve spatial structure for visualization
    """
    # Use clean encoder config with intermediate_layers=None to disable them
    dpsk_config = EncoderConfig(
        encoder_type=EncoderType.DPSK,
        model_path="deepseek-ai/DeepSeek-OCR",
        intermediate_layers=None,  # None disables intermediate layers (not [])
        remove_separators=True,
    )

    extractor = FeatureExtractor(encoder_configs=[dpsk_config])

    specs = [
        FeatureSpec('sam_40x40', 'final', 'sam_raw', 'none'),
        FeatureSpec('sam_pooled', 'final', 'sam_raw', 'mean'),
        FeatureSpec('dpsk_final', 'final', 'visual', 'none'),  # Keep spatial structure
    ]

    results_raw = extractor.extract_multiple(
        image,
        [EncoderType.DPSK],
        specs
    )

    # Convert (encoder_type, spec_name) keys to just spec_name
    results = {spec.name: results_raw[(EncoderType.DPSK, spec.name)] for spec in specs
               if (EncoderType.DPSK, spec.name) in results_raw}

    extractor.cleanup()
    return results


def extract_layer_features(
    image: Union[Path, Image.Image],
    encoder_type: EncoderType = EncoderType.DPSK,
) -> Dict[str, FeatureData]:
    """Extract features from multiple layers.

    Returns:
        Dict mapping spec_name to FeatureData (string keys for convenience)
    """
    extractor = FeatureExtractor()

    if encoder_type == EncoderType.DPSK:
        specs = [
            FeatureSpec('final', 'final', 'visual', 'mean'),
            FeatureSpec('layer_0', 'intermediate_0', 'visual', 'mean'),
            FeatureSpec('layer_1', 'intermediate_1', 'visual', 'mean'),
            FeatureSpec('layer_2', 'intermediate_2', 'visual', 'mean'),
        ]
    else:  # Qwen3VL
        specs = [
            FeatureSpec('final', 'final', 'visual', 'mean'),
            FeatureSpec('deepstack_0', 'deepstack_0', 'visual', 'mean'),
            FeatureSpec('deepstack_1', 'deepstack_1', 'visual', 'mean'),
            FeatureSpec('deepstack_2', 'deepstack_2', 'visual', 'mean'),
        ]

    results_raw = extractor.extract_multiple(
        image,
        [encoder_type],
        specs
    )

    # Convert (encoder_type, spec_name) keys to just spec_name
    results = {spec.name: results_raw[(encoder_type, spec.name)] for spec in specs
               if (encoder_type, spec.name) in results_raw}

    extractor.cleanup()
    return results
