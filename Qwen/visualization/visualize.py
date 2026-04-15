#!/usr/bin/env python3
"""
Unified visualization CLI for OCR features.

This script provides a centralized interface for all visualization tasks:
- Single image feature visualization
- Batch processing for transparent eval data
- SAM feature extraction and visualization
- Multi-layer feature comparisons
- t-SNE/UMAP embeddings

Usage:
    python -m Qwen.visualization.visualize single-image <image_path>
    python -m Qwen.visualization.visualize sam-comparison <image_path>
    python -m Qwen.visualization.visualize batch-sam <metadata_path>
    python -m Qwen.visualization.visualize layers <image_path> --encoder dpsk
    python -m Qwen.visualization.visualize embeddings <features.pkl>
"""

import argparse
import json
import pickle
import sys
import traceback
from project_paths import get_deepseek_ocr_dir
from pathlib import Path

import numpy as np
from PIL import Image

from .vis_core import EncoderType, setup_logging
from .vis_features import (
    FeatureSpec,
    FeatureExtractor,
    extract_layer_features,
)
from .vis_plots import (
    BatchVisualizer,
    EmbeddingVisualizer,
    SingleImageVisualizer,
    plot_sam_single_image,
)


def cmd_single_image(args):
    """Visualize features for a single image."""
    logger = setup_logging(args.verbose)

    logger.info(f"Visualizing features for: {args.image}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load image
    image = Image.open(args.image).convert('RGB')

    # Extract features based on encoder
    if args.encoder == 'dpsk':
        encoder_type = EncoderType.DPSK
        specs = [
            FeatureSpec('Final', 'final', 'visual', 'none'),
            FeatureSpec('Layer 6', 'intermediate_0', 'visual', 'none'),
            FeatureSpec('Layer 12', 'intermediate_1', 'visual', 'none'),
            FeatureSpec('Layer 18', 'intermediate_2', 'visual', 'none'),
        ]
    else:  # qwen
        encoder_type = EncoderType.QWEN3VL
        specs = [
            FeatureSpec('Final', 'final', 'visual', 'none'),
            FeatureSpec('DS0', 'deepstack_0', 'visual', 'none'),
            FeatureSpec('DS1', 'deepstack_1', 'visual', 'none'),
            FeatureSpec('DS2', 'deepstack_2', 'visual', 'none'),
        ]

    extractor = FeatureExtractor()
    features_dict = {}

    for spec in specs:
        try:
            result = extractor.extract(args.image, encoder_type, spec)
            # Convert to spatial format for visualization
            if encoder_type == EncoderType.DPSK:
                # 100 tokens -> 10x10 grid
                features = result.features.reshape(10, 10, -1)
            else:
                # Qwen tokens vary - approximate grid
                n_tokens = result.features.shape[0]
                grid_size = int(n_tokens ** 0.5)
                if grid_size * grid_size == n_tokens:
                    features = result.features.reshape(grid_size, grid_size, -1)
                else:
                    # Use as-is (will show as text)
                    features = result.features

            features_dict[spec.name] = features
            logger.info(f"  ✓ Extracted {spec.name}: {features.shape}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to extract {spec.name}: {e}")

    extractor.cleanup()

    # Create visualization
    vis = SingleImageVisualizer(Path(args.output_dir))
    output_path = vis.plot_feature_comparison(
        image,
        features_dict,
        title=f"{args.encoder.upper()} Features: {Path(args.image).name}",
        output_name=f"{Path(args.image).stem}_features.png",
    )

    logger.info(f"✓ Saved visualization to: {output_path}")


def cmd_sam_comparison(args):
    """Create SAM feature pipeline visualization."""
    logger = setup_logging(args.verbose)

    logger.info(f"Creating SAM pipeline visualization for: {args.image}")

    output_path = plot_sam_single_image(
        Path(args.image),
        Path(args.output_dir),
    )

    logger.info(f"✓ Saved SAM comparison to: {output_path}")


def cmd_batch_sam(args):
    """Batch SAM visualization for transparent eval data."""
    logger = setup_logging(args.verbose)

    # Load metadata
    logger.info(f"Loading metadata from: {args.metadata or 'default metadata file'}")
    repo_root = get_deepseek_ocr_dir()
    metadata_path = repo_root / "OCRVL/llamafactory/data/ocrvl_transparent_eval.metadata.json"

    if args.metadata:
        metadata_path = Path(args.metadata)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    samples = metadata['samples']
    logger.info(f"  ✓ Loaded {len(samples)} samples")

    # Initialize extractor
    extractor = FeatureExtractor()

    # Initialize visualizer
    vis = BatchVisualizer(Path(args.output_dir))

    # Process each sample
    for sample in samples:
        sample_id = sample['id']
        task = sample['task']
        logger.info(f"\n[{sample['index']+1}/{len(samples)}] {sample_id} ({task})")

        for img_idx, img_rel_path in enumerate(sample['images']):
            img_path = repo_root / img_rel_path
            if not img_path.exists():
                logger.warning(f"  ⚠ Image not found: {img_path}")
                continue

            try:
                # Load image
                image = Image.open(img_path).convert('RGB')

                # Extract SAM and DPSK features
                sam_spec = FeatureSpec('sam_40x40', 'final', 'sam_raw', 'none')
                dpsk_spec = FeatureSpec('dpsk_final', 'final', 'visual', 'none')

                sam_result = extractor.extract(img_path, EncoderType.DPSK, sam_spec)
                dpsk_result = extractor.extract(img_path, EncoderType.DPSK, dpsk_spec)

                # Create visualization
                task_output_dir = Path(args.output_dir) / task
                task_output_dir.mkdir(parents=True, exist_ok=True)

                single_vis = SingleImageVisualizer(task_output_dir)
                output_path = single_vis.plot_sam_pipeline(
                    image,
                    sam_result.features,
                    dpsk_result.features.reshape(10, 10, -1) if dpsk_result.features.ndim == 1 else dpsk_result.features,
                    title=f"{sample_id} | {task} (Image {img_idx+1}/2)" if len(sample['images']) == 2 else f"{sample_id} | {task}",
                    output_name=f"{sample_id}_img{img_idx}.png",
                )

                logger.info(f"  ✓ {output_path.relative_to(Path(args.output_dir))}")

            except Exception as e:
                logger.warning(f"  ✗ Failed: {e}")
                traceback.print_exc()
                continue

    extractor.cleanup()
    logger.info(f"\n✓ Batch SAM visualization complete!")
    logger.info(f"  Output directory: {args.output_dir}")


def cmd_layers(args):
    """Visualize features from multiple layers."""
    logger = setup_logging(args.verbose)

    logger.info(f"Extracting layer features from: {args.image}")
    logger.info(f"Encoder: {args.encoder}")

    # Extract layer features
    encoder_type = EncoderType.DPSK if args.encoder == 'dpsk' else EncoderType.QWEN3VL
    features_dict = extract_layer_features(args.image, encoder_type)

    # Convert to spatial format
    logger.info("Creating visualization...")

    # Load image
    image = Image.open(args.image).convert('RGB')

    # Prepare features for plotting
    plot_features = {}
    for key, result in features_dict.items():
        if encoder_type == EncoderType.DPSK:
            # 100 tokens -> 10x10 grid
            features = result.features.reshape(10, 10, -1) if result.features.ndim == 2 else result.features
        else:
            # Qwen - approximate grid
            features = result.features
            if features.ndim == 2:
                n_tokens = features.shape[0]
                grid_size = int(n_tokens ** 0.5)
                if grid_size * grid_size == n_tokens:
                    features = features.reshape(grid_size, grid_size, -1)

        plot_features[key] = features
        logger.info(f"  {key}: {features.shape}")

    # Create visualization
    vis = SingleImageVisualizer(Path(args.output_dir))
    output_path = vis.plot_feature_comparison(
        image,
        plot_features,
        title=f"{args.encoder.upper()} Layer Features: {Path(args.image).name}",
        output_name=f"{Path(args.image).stem}_layers.png",
    )

    logger.info(f"✓ Saved visualization to: {output_path}")


def cmd_embeddings(args):
    """Create t-SNE/UMAP visualizations from feature file."""
    logger = setup_logging(args.verbose)

    logger.info(f"Loading features from: {args.features}")

    with open(args.features, 'rb') as f:
        data = pickle.load(f)

    # Extract features and labels
    if 'dpsk_features' in data:
        features = data['dpsk_features']['final_visual']
        categories = np.array(data['categories'])
    else:
        raise ValueError("Feature file format not recognized")

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Categories: {np.unique(categories)}")

    # Create visualizations
    vis = EmbeddingVisualizer(Path(args.output_dir))

    if args.method in ['tsne', 'both']:
        logger.info("Computing t-SNE...")
        vis.plot_tsne(
            features,
            categories,
            title="DPSK Final Visual Features - t-SNE",
            output_name="dpsk_final_tsne.png",
            perplexity=args.perplexity,
        )
        logger.info("  ✓ t-SNE complete")

    if args.method in ['umap', 'both']:
        logger.info("Computing UMAP...")
        vis.plot_umap(
            features,
            categories,
            title="DPSK Final Visual Features - UMAP",
            output_name="dpsk_final_umap.png",
        )
        logger.info("  ✓ UMAP complete")

    logger.info(f"✓ Embedding visualizations saved to: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified visualization CLI for OCR features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize features for a single image
  python -m Qwen.visualization.visualize single-image image.jpg --encoder dpsk

  # SAM feature pipeline comparison
  python -m Qwen.visualization.visualize sam-comparison image.jpg

  # Batch SAM visualization for transparent eval
  python -m Qwen.visualization.visualize batch-sam --metadata ocrvl_transparent_eval.metadata.json

  # Multi-layer feature visualization
  python -m Qwen.visualization.visualize layers image.jpg --encoder qwen

  # t-SNE/UMAP embeddings from features file
  python -m Qwen.visualization.visualize embeddings features.pkl --method both
        """
    )

    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output-dir', '-o', default='results/feature_vis/plots', help='Output directory')

    subparsers = parser.add_subparsers(dest='command', help='Visualization command')

    # single-image command
    single_parser = subparsers.add_parser('single-image', help='Visualize features for a single image')
    single_parser.add_argument('image', type=Path, help='Path to image')
    single_parser.add_argument('--encoder', choices=['dpsk', 'qwen'], default='dpsk', help='Encoder to use')

    # sam-comparison command
    sam_parser = subparsers.add_parser('sam-comparison', help='SAM feature pipeline visualization')
    sam_parser.add_argument('image', type=Path, help='Path to image')

    # batch-sam command
    batch_parser = subparsers.add_parser('batch-sam', help='Batch SAM visualization')
    batch_parser.add_argument('--metadata', type=Path, help='Path to metadata JSON')

    # layers command
    layers_parser = subparsers.add_parser('layers', help='Multi-layer feature visualization')
    layers_parser.add_argument('image', type=Path, help='Path to image')
    layers_parser.add_argument('--encoder', choices=['dpsk', 'qwen'], default='dpsk', help='Encoder to use')

    # embeddings command
    embed_parser = subparsers.add_parser('embeddings', help='t-SNE/UMAP visualizations')
    embed_parser.add_argument('features', type=Path, help='Path to features.pkl file')
    embed_parser.add_argument('--method', choices=['tsne', 'umap', 'both'], default='tsne', help='Embedding method')
    embed_parser.add_argument('--perplexity', type=int, default=30, help='t-SNE perplexity')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to command handler
    commands = {
        'single-image': cmd_single_image,
        'sam-comparison': cmd_sam_comparison,
        'batch-sam': cmd_batch_sam,
        'layers': cmd_layers,
        'embeddings': cmd_embeddings,
    }

    handler = commands.get(args.command)
    if handler:
        try:
            handler(args)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
