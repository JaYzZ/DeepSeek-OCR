#!/usr/bin/env python3
"""
Generate t-SNE and UMAP visualizations for DeepSeek OCR and Qwen3VL features.

This script:
1. Loads extracted features from both encoders
2. Computes t-SNE and UMAP embeddings for:
   - DeepSeek OCR CLS tokens + visual tokens (optional)
   - Qwen3VL visual output + deepstack features
3. Creates comparison plots

Output: ./results/feature_vis/plots/tsne/ and ./results/feature_vis/plots/umap/
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns
from typing import Dict, List
import argparse

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class FeatureVisualizer:
    """Create t-SNE and UMAP visualizations for vision features."""

    def __init__(self, features_path: Path, output_dir: Path, use_cls: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cls = use_cls  # Whether to visualize CLS tokens (default: True)

        # Always use cuda:0
        torch.cuda.set_device(0)
        print(f"Using GPU: cuda:0 (forced)")

        # Load features
        print(f"\nLoading features from {features_path}...")
        with open(features_path, 'rb') as f:
            self.data = pickle.load(f)

        self.dpsk_features = self.data['dpsk_features']
        self.qwen_features = self.data['qwen_features']
        self.categories = np.array(self.data['categories'])
        self.filenames = self.data['filenames']

        print(f"✓ Loaded {len(self.categories)} samples")
        print(f"  Categories: {np.unique(self.categories).tolist()}")
        print(f"  Visualize CLS tokens: {self.use_cls}")
        print(f"  Device: {self.device}")

        # Color map for categories
        unique_cats = sorted(np.unique(self.categories))
        self.category_colors = {
            cat: sns.color_palette("husl", len(unique_cats))[i]
            for i, cat in enumerate(unique_cats)
        }

    def compute_tsne(self, features: np.ndarray, perplexity: int = 30, n_iter: int = 1000) -> np.ndarray:
        """Compute t-SNE embedding."""
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Compute t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42,
            verbose=1,
        )
        embedding = tsne.fit_transform(features_scaled)

        return embedding

    def compute_umap(self, features: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """Compute UMAP embedding."""
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Compute UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',  # Use cosine similarity for text features
            random_state=42,
        )
        embedding = reducer.fit_transform(features_scaled)

        return embedding

    def plot_embedding(
        self,
        embedding: np.ndarray,
        title: str,
        filename: str,
        method: str = "t-SNE",
        show_legend: bool = True,
    ):
        """Create scatter plot for 2D embedding."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot each category
        for category in sorted(np.unique(self.categories)):
            mask = self.categories == category
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[self.category_colors[caption for caption in [self.category_colors] if caption == category]],
                label=category,
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

        ax.set_title(f"{title} ({method})", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
        ax.set_ylabel(f'{method} Dimension 2', fontsize=12)

        if show_legend:
            ax.legend(
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True,
                fontsize=10,
            )

        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{method}/{filename}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_comparison(
        self,
        dpsk_embedding: np.ndarray,
        qwen_embedding: np.ndarray,
        title: str,
        filename: str,
        method: str = "t-SNE",
    ):
        """Create side-by-side comparison plot."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # DeepSeek OCR
        for category in sorted(np.unique(self.categories)):
            mask = self.categories == category
            axes[0].scatter(
                dpsk_embedding[mask, 0],
                dpsk_embedding[mask, 1],
                c=[self.category_colors[caption] for caption in [self.category_colors if caption == category]],
                label=category,
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

        axes[0].set_title('DeepSeek OCR', fontsize=14, fontweight='bold')
        axes[0].set_xlabel(f'{method} Dimension 1', fontsize=11)
        axes[0].set_ylabel(f'{method} Dimension 2', fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Qwen3VL
        for category in sorted(np.unique(self.categories)):
            mask = self.categories == category
            axes[1].scatter(
                qwen_embedding[mask, 0],
                qwen_embedding[mask, 1],
                c=[self.category_colors[caption] for caption in [self.category_colors if caption == category]],
                label=category,
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

        axes[1].set_title('Qwen3VL', fontsize=14, fontweight='bold')
        axes[1].set_xlabel(f'{method} Dimension 1', fontsize=11)
        axes[1].set_ylabel(f'{method} Dimension 2', fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, -0.05),
            loc='upper center',
            ncol=len(np.unique(self.categories)),
            frameon=True,
            fontsize=10,
        )

        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{method}/{filename}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def visualize_all(self, method: str = "tsne"):
        """Generate all visualizations for the specified method."""
        print(f"\n{'='*60}")
        print(f"Generating {method.upper()} Visualizations")
        print(f"{'='*60}\n")

        # Determine which features to visualize based on use_cls
        token_type = "CLS" if self.use_cls else "Visual"
        print(f"Visualizing {token_type} tokens for DeepSeek OCR")
        print(f"Using device: {self.device}")

        # 1. Final layer features (CLS or visual)
        if self.use_cls and 'final_cls' in self.dpsk_features:
            print(f"\n1. Computing {method.upper()} for final CLS token...")
            dpsk_final_embedding = self.compute_embedding(self.dpsk_features['final_cls'], method=method, n_neighbors=15)
            self.plot_embedding(
                dpsk_final_embedding,
                "DeepSeek OCR - Final CLS Token (CLIP Layer 24)",
                "final_cls",
                method=method
            )

        elif 'final_visual' in self.dpsk_features:
            print(f"\n1. Computing {method.upper()} for final visual tokens...")
            dpsk_visual_embedding = self.compute_embedding(self.dpsk_features['final_visual'], method=method, n_neighbors=15)

        # Always compute visual for comparison with Qwen
        if 'final_visual' in self.dpsk_features:
            if 'final_visual' not in locals():
                dpsk_visual_embedding = self.compute_embedding(self.dpsk_features['final_visual'], method=method, n_neighbors=15)
            qwen_visual_embedding = self.compute_embedding(self.qwen_features['visual_output'], method=method, n_neighbors=15)

            print("\n  Creating plots...")
            self.plot_embedding(
                dpsk_visual_embedding,
                "DeepSeek OCR - Final Visual Tokens",
                "final_visual",
                method=method
            )
            self.plot_embedding(
                qwen_visual_embedding,
                "Qwen3VL - Visual Output Features (Final Layer)",
                "qwen_visual_output",
                method=method
            )
            self.plot_comparison(
                dpsk_visual_embedding,
                qwen_visual_embedding,
                "Visual Output Features Comparison",
                "visual_output_comparison",
                method=method
            )

        # 2. Intermediate layers (CLS or visual tokens)
        layer_names = ['layer_0', 'layer_1', 'layer_2']
        clip_layers = ['CLIP Layer 6', 'CLIP Layer 12', 'CLIP Layer 18']

        for i, (layer_name, clip_layer) in enumerate(zip(layer_names, clip_layers)):
            cls_key = f'{layer_name}_cls'
            visual_key = f'{layer_name}_visual'

            # Choose which feature to use based on use_cls
            if self.use_cls and cls_key in self.dpsk_features:
                key = cls_key
                key_type = "CLS Token"
            elif visual_key in self.dpsk_features:
                key = visual_key
                key_type = "Visual Tokens (attention-weighted)"
            else:
                continue

            print(f"\n{i+2}. Computing {method.upper()} for {layer_name} ({key_type})...")
            dpsk_embedding = self.compute_embedding(self.dpsk_features[key], method=method, n_neighbors=15)

            # Get corresponding Qwen deepstack layer
            qwen_key = f'deepstack_{i}'
            if qwen_key in self.qwen_features:
                qwen_embedding = self.compute_embedding(self.qwen_features[qwen_key], method=method, n_neighbors=15)

                print("\n  Creating plots...")
                self.plot_embedding(
                    dpsk_embedding,
                    f"DeepSeek OCR - {key_type} ({clip_layer})",
                    f"dpsk_{layer_name}",
                    method=method
                )
                self.plot_embedding(
                    qwen_embedding,
                    f"Qwen3VL - Deepstack Layer {i}",
                    f"qwen_deepstack_{i}",
                    method=method
                )
                self.plot_comparison(
                    dpsk_embedding,
                    qwen_embedding,
                    f"Layer {i} Comparison ({clip_layer})",
                    f"{layer_name}_comparison",
                    method=method
                )

        # 3. Combined visualization - all layers
        print(f"\n{6}. Creating combined multi-layer visualization...")
        self._plot_all_layers(method=method)

        print(f"\n{'='*60}")
        print(f"✓ All {method.upper()} visualizations saved to {self.output_dir}/{method.lower()}/")
        print(f"{'='*60}\n")

    def _plot_all_layers(self, method: str = "tsne"):
        """Create a grid showing all layers for both models."""
        # Prepare data for DeepSeek OCR
        token_type = "cls" if self.use_cls else "visual"
        suffix = f"_{token_type}"

        if self.use_cls and 'final_cls' in self.dpsk_features:
            # Visualize CLS tokens: final_cls + intermediate CLS tokens
            dpsk_layers = [self.dpsk_features['final_cls']]
            layer_names = ['Final\n(CLP\n24)']
        else:
            # Visualize visual tokens
            dpsk_layers = [self.dpsk_features['final_visual']]
            layer_names = ['Final\n(No\nCLS)']

        for i in range(3):
            key = f'layer_{i}{suffix}'
            if key in self.dpsk_features:
                dpsk_layers.append(self.dpsk_features[key])
                layer_names.append(f'Layer {i}\n(CLP {[6,12,18][i]})')

        # Qwen features (always visual)
        qwen_layers = [self.qwen_features['visual_output']]
        for i in range(3):
            key = f'deepstack_{i}'
            if key in self.qwen_features:
                qwen_layers.append(self.qwen_features[key])

        n_layers = len(dpsk_layers)
        fig, axes = plt.subplots(2, n_layers, figsize=(6*n_layers, 12))

        for col in range(n_layers):
            # DeepSeek row
            print(f"  Computing {method.upper()} for DeepSeek layer {col}...")
            dpsk_embedding = self.compute_embedding(
                dpsk_layers[col],
                method=method,
                n_neighbors=min(15, len(self.categories)-1) if method == "umap" else min(30, len(self.categories)-1)
            )

            for category in sorted(np.unique(self.categories)):
                mask = self.categories == category
                axes[0, col].scatter(
                    dpsk_embedding[mask, 0],
                    dpsk_embedding[mask, 1],
                    c=[self.category_colors[caption] for caption in [self.category_colors if caption == category]],
                    label=category,
                    alpha=0.7,
                    s=30,
                    edgecolors='white',
                    linewidth=0.3,
                )

            axes[0, col].set_title(f'DeepSeek OCR\n{layer_names[col]}', fontsize=11, fontweight='bold')
            axes[0, col].grid(True, alpha=0.3)
            if col == 0:
                axes[0, col].set_ylabel(f'{method.upper()} Dim 2', fontsize=10)

            # Qwen row
            if col < len(qwen_layers):
                print(f"  Computing {method.upper()} for Qwen layer {col}...")
                qwen_embedding = self.compute_embedding(
                    qwen_layers[col],
                    method=method,
                    n_neighbors=min(15, len(self.categories)-1) if method == "umap" else min(30, len(self.categories)-1)
                )

                for category in sorted(np.unique(self.categories)):
                    mask = self.categories == category
                    axes[1, col].scatter(
                        qwen_embedding[mask, 0],
                        qwen_embedding[mask, 1],
                        c=[self.category_colors[caption] for caption in [self.category_colors if caption == category]],
                        label=category,
                        alpha=0.7,
                        s=30,
                        edgecolors='white',
                        linewidth=0.3,
                    )

                axes[1, col].set_title(f'Qwen3VL\n{layer_names[col]}', fontsize=11, fontweight='bold')
            else:
                # Hide empty subplot
                axes[1, col].axis('off')

            axes[1, col].set_xlabel(f'{method.upper()} Dim 1', fontsize=10)
            axes[1, col].grid(True, alpha=0.3)
            if col == 0:
                axes[1, col].set_ylabel(f'{method.upper()} Dim 2', fontsize=10)

        # Shared legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            bbox_to_anchor=(0.5, -0.02),
            loc='upper center',
            ncol=len(np.unique(self.categories)),
            frameon=True,
            fontsize=9,
        )

        token_type_str = "CLS Tokens" if self.use_cls else "Visual Tokens"
        fig.suptitle(
            f'Multi-Layer Feature Comparison: DeepSeek OCR ({token_type_str}) vs Qwen3VL ({method.upper()})',
            fontsize=14,
            fontweight='bold',
            y=0.995,
        )
        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"all_layers_comparison.{method}"
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate t-SNE and UMAP visualizations for extracted features')
    parser.add_argument(
        '--use-cls',
        action='store_true',
        default=True,
        help='Visualize CLS tokens (default: True). Use --no-use-cls to visualize visual tokens instead.'
    )
    parser.add_argument(
        '--no-use-cls',
        dest='use_cls',
        action='store_false',
        help='Visualize visual tokens instead of CLS tokens'
    )
    parser.add_argument(
        '--method',
        choices=['tsne', 'umap', 'both'],
        default='tsne',
        help='Visualization method: tsne, umap, or both'
    )
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Device to use for computations'
    )

    args = parser.parse_args()

    # Parse device
    device = args.device.strip()
    if device.startswith("cuda"):
        device_idx = device.replace("cuda", "")
        if not device_idx.isdigit():
            print(f"✗ Error: Invalid device '{device}'. Use 'cuda:0', 'cuda:1', etc.")
            sys.exit(1)
    device = f"cuda:{device_idx}"

    print(f"Using device: {device}")

    # Paths
    base_dir = Path(__file__).parent
    features_path = base_dir / "results" / "feature_vis" / "features" / "features.pkl"
    output_dir = base_dir / "results" / "feature_vis" / "plots"

    # Check if features exist
    if not features_path.exists():
        print(f"Error: Please run extract_features.py first!")
        print(f"  Expected: {features_path}")
        sys.exit(1)

    # Create visualizations
    if args.method == "umap":
        visualizer = FeatureVisualizer(features_path, output_dir, use_cls=args.use_cls, device=device)
        visualizer.visualize_all(method="umap")
    elif args.method == "tsne":
        visualizer = FeatureVisualizer(features_path, output_dir, use_cls=args.use_cls, device=device)
        visualizer.visualize_all(method="tsne")
    else:  # both
        visualizer = FeatureVisualizer(features_path, output_dir, use_cls=args.use_cls, device=device)

        print("\n")
        print("="*60)
        print("✓ All visualizations complete!")
        print("="*60)
        print(f"\nGenerated plots:")
        for plot_dir in [output_dir / "tsne", output_dir / "umap"]:
            plot_dir.mkdir(parents=True, exist_ok=True)
            for plot_file in sorted(plot_dir.glob("*comparison.png")):
                print(f"  {plot_dir.name}/{plot_file.name}")

        print(f"\nTo view images:")
        print(f"  Linux: xdg-open {output_dir}/tsne/*.png")
        print(f"  Windows: start {output_dir}\\tsne\\*.png")
        print("="*60)
