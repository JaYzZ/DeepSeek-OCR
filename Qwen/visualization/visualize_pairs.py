#!/usr/bin/env python3
"""
Generate pairwise t-SNE visualizations for natural images vs rendered text.

This script:
1. Loads extracted features and metadata with pair information
2. Computes t-SNE/UMAP embeddings
3. Creates plots showing natural vs rendered pairs with different markers
4. Optionally draws connecting lines between pairs

Output: ./results/feature_vis/plots/pairs/
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class PairwiseVisualizer:
    """Create t-SNE/UMAP visualizations for natural vs rendered text pairs."""

    def __init__(self, features_path: Path, metadata_path: Path, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load metadata
        print(f"\nLoading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load features
        print(f"Loading features from {features_path}...")
        with open(features_path, 'rb') as f:
            feature_data = pickle.load(f)

        self.dpsk_features = feature_data['dpsk_features']
        self.qwen_features = feature_data['qwen_features']
        self.categories = np.array(feature_data['categories'])
        self.filenames = np.array(feature_data['filenames'])

        # Build type and pair_id arrays from metadata
        self.types = []
        self.pair_ids = []
        filename_to_meta = {m['filename']: m for m in self.metadata}

        for fname in self.filenames:
            meta = filename_to_meta.get(fname, {})
            self.types.append(meta.get('type', 'unknown'))
            self.pair_ids.append(meta.get('pair_id', -1))

        self.types = np.array(self.types)
        self.pair_ids = np.array(self.pair_ids)

        print(f"✓ Loaded {len(self.categories)} samples")
        print(f"  Categories: {np.unique(self.categories).tolist()}")
        print(f"  Types: {np.unique(self.types).tolist()}")

        # Color map for categories
        unique_cats = sorted(np.unique(self.categories))
        self.category_colors = {
            cat: sns.color_palette("husl", len(unique_cats))[i]
            for i, cat in enumerate(unique_cats)
        }

        # Marker styles for types
        self.type_markers = {
            'natural': 'o',      # circle
            'rendered_text': 'X',  # X marker
            'unknown': 's'       # square
        }

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {self.device}")

    def compute_embedding(
        self, features: np.ndarray, method: str = "tsne", **kwargs
    ) -> np.ndarray:
        """Compute t-SNE or UMAP embedding."""
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        if method == "tsne":
            perplexity = kwargs.get('perplexity', 30)
            embedder = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=42,
                verbose=1,
            )
        else:  # umap
            n_neighbors = kwargs.get('n_neighbors', 15)
            min_dist = kwargs.get('min_dist', 0.1)
            embedder = umap.UMAP(
                n_components=2,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='cosine',
                random_state=42,
            )

        return embedder.fit_transform(features_scaled)

    def plot_pairs_by_category(
        self,
        embedding: np.ndarray,
        features_name: str,
        method: str = "tsne",
        draw_lines: bool = False,
    ):
        """Plot embedding with natural vs rendered pairs, grouped by category."""
        fig, ax = plt.subplots(figsize=(14, 10))

        # For each category, plot natural and rendered separately
        for category in sorted(np.unique(self.categories)):
            cat_mask = self.categories == category
            color = self.category_colors[category]

            # Plot natural images (circles)
            natural_mask = cat_mask & (self.types == 'natural')
            if natural_mask.sum() > 0:
                ax.scatter(
                    embedding[natural_mask, 0],
                    embedding[natural_mask, 1],
                    c=[color],
                    marker=self.type_markers['natural'],
                    label=f'{category} (natural)',
                    alpha=0.7,
                    s=80,
                    edgecolors='white',
                    linewidth=0.5,
                )

            # Plot rendered text (X markers)
            rendered_mask = cat_mask & (self.types == 'rendered_text')
            if rendered_mask.sum() > 0:
                ax.scatter(
                    embedding[rendered_mask, 0],
                    embedding[rendered_mask, 1],
                    c=[color],
                    marker=self.type_markers['rendered_text'],
                    label=f'{category} (rendered)',
                    alpha=0.7,
                    s=80,
                    edgecolors='white',
                    linewidth=0.5,
                )

        # Draw connecting lines between pairs
        if draw_lines:
            self._draw_pair_lines(ax, embedding)

        ax.set_title(
            f"{features_name} - Natural vs Rendered Text ({method.upper()})",
            fontsize=14, fontweight='bold', pad=15
        )
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Custom legend
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True,
            fontsize=8,
            ncol=1,
        )

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{method}/{features_name}_pairs_by_category.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_pairs_overlay(
        self,
        embedding: np.ndarray,
        features_name: str,
        method: str = "tsne",
        draw_lines: bool = True,
        max_lines: int = 20,
    ):
        """Plot with overlay showing pairs connected by lines."""
        fig, ax = plt.subplots(figsize=(14, 10))

        # Plot all points
        for img_type in ['natural', 'rendered_text']:
            type_mask = self.types == img_type
            if type_mask.sum() == 0:
                continue

            # Color by category
            for category in sorted(np.unique(self.categories)):
                mask = type_mask & (self.categories == category)
                if mask.sum() == 0:
                    continue

                color = self.category_colors[category]
                ax.scatter(
                    embedding[mask, 0],
                    embedding[mask, 1],
                    c=[color],
                    marker=self.type_markers[img_type],
                    s=80 if img_type == 'natural' else 100,
                    alpha=0.7,
                    edgecolors='white',
                    linewidth=0.5,
                )

        # Draw lines for a subset of pairs (avoid clutter)
        if draw_lines:
            self._draw_pair_lines(ax, embedding, max_lines=max_lines)

        # Create legend elements
        legend_elements = []
        # Category colors
        for cat in sorted(np.unique(self.categories)):
            legend_elements.append(
                Patch(facecolor=self.category_colors[cat], label=cat, alpha=0.7)
            )
        # Markers
        legend_elements.extend([
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                   markersize=10, label='Natural image', markeredgecolor='white'),
            Line2D([0], [0], marker='X', color='w', markerfacecolor='gray',
                   markersize=10, label='Rendered text', markeredgecolor='white'),
        ])

        ax.set_title(
            f"{features_name} - Pair Connections ({method.upper()})",
            fontsize=14, fontweight='bold', pad=15
        )
        ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=11)
        ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=11)
        ax.grid(True, alpha=0.3)

        ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True,
            fontsize=9,
        )

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{method}/{features_name}_pairs_overlay.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def plot_distance_analysis(
        self,
        embedding: np.ndarray,
        features_name: str,
        method: str = "tsne",
    ):
        """Analyze and plot distances between paired items."""
        # Compute distances for each pair
        pair_distances = {}
        pair_indices = {}

        for cat in np.unique(self.categories):
            cat_distances = []
            cat_indices = []

            for pair_id in np.unique(self.pair_ids):
                if pair_id == -1:
                    continue

                # Find indices for this pair
                mask = (self.pair_ids == pair_id) & (self.categories == cat)
                indices = np.where(mask)[0]

                if len(indices) == 2:
                    # Get types
                    types = self.types[indices]
                    # Ensure one is natural and one is rendered
                    if set(types) == {'natural', 'rendered_text'}:
                        idx_natural = indices[self.types[indices] == 'natural'][0]
                        idx_rendered = indices[self.types[indices] == 'rendered_text'][0]

                        # Compute Euclidean distance in embedding space
                        dist = np.linalg.norm(
                            embedding[idx_natural] - embedding[idx_rendered]
                        )
                        cat_distances.append(dist)
                        cat_indices.append((idx_natural, idx_rendered))

            if cat_distances:
                pair_distances[cat] = cat_distances
                pair_indices[cat] = cat_indices

        # Plot distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = sorted(pair_distances.keys())
        positions = []
        data = []
        colors = []

        for i, cat in enumerate(categories):
            distances = pair_distances[cat]
            positions.extend([i] * len(distances))
            data.extend(distances)
            colors.extend([self.category_colors[cat]] * len(distances))

        # Violin plot
        parts = ax.violinplot(
            [pair_distances[cat] for cat in categories],
            positions=range(len(categories)),
            showmeans=True,
            showmedians=True,
        )

        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.category_colors[categories[i]])
            pc.set_alpha(0.7)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.set_ylabel(f'{method.upper()} Distance', fontsize=11)
        ax.set_title(
            f'{features_name} - Distribution of Pair Distances by Category',
            fontsize=13, fontweight='bold'
        )
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        # Save
        output_path = self.output_dir / f"{method}/{features_name}_pair_distances.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

        # Print statistics
        print(f"\n  Pair distance statistics ({method.upper()}):")
        for cat in categories:
            dists = pair_distances[cat]
            print(f"    {cat:15s}: mean={np.mean(dists):.3f}, std={np.std(dists):.3f}, "
                  f"min={np.min(dists):.3f}, max={np.max(dists):.3f}")

    def _draw_pair_lines(self, ax, embedding: np.ndarray, max_lines: int = None):
        """Draw lines connecting paired items."""
        drawn = 0
        for pair_id in np.unique(self.pair_ids):
            if pair_id == -1:
                continue
            if max_lines and drawn >= max_lines:
                break

            mask = self.pair_ids == pair_id
            indices = np.where(mask)[0]

            if len(indices) == 2:
                types = self.types[indices]
                if set(types) == {'natural', 'rendered_text'}:
                    # Get category color
                    cat = self.categories[indices][0]
                    color = self.category_colors[cat]

                    ax.plot(
                        embedding[indices, 0],
                        embedding[indices, 1],
                        c=color,
                        alpha=0.3,
                        linewidth=1,
                        linestyle='--',
                    )
                    drawn += 1

    def visualize_all(self, method: str = "tsne", draw_lines: bool = True):
        """Generate all pairwise visualizations."""
        print(f"\n{'='*60}")
        print(f"Generating {method.upper()} Pairwise Visualizations")
        print(f"{'='*60}\n")

        # Feature sets to visualize
        feature_sets = [
            ('dpsk', 'final_visual', self.dpsk_features),
            ('qwen', 'visual_output', self.qwen_features),
        ]

        for prefix, feat_name, features in feature_sets:
            if feat_name not in features:
                print(f"  Skipping {feat_name} (not found)")
                continue

            print(f"\n{prefix.upper()}: {feat_name}")
            feat_array = np.array(features[feat_name])
            print(f"  Feature shape: {feat_array.shape}")

            # Compute embedding
            embedding = self.compute_embedding(feat_array, method=method)

            # Generate plots
            print("\n  Generating plots...")
            self.plot_pairs_by_category(
                embedding, f"{prefix}_{feat_name}", method, draw_lines=False
            )
            self.plot_pairs_overlay(
                embedding, f"{prefix}_{feat_name}", method, draw_lines=draw_lines
            )
            self.plot_distance_analysis(
                embedding, f"{prefix}_{feat_name}", method
            )

        print(f"\n{'='*60}")
        print(f"✓ All visualizations saved to {self.output_dir}/{method.lower()}/")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate pairwise t-SNE/UMAP visualizations')
    parser.add_argument(
        '--method',
        choices=['tsne', 'umap', 'both'],
        default='tsne',
        help='Visualization method'
    )
    parser.add_argument(
        '--draw-lines',
        action='store_true',
        default=True,
        help='Draw connecting lines between pairs'
    )
    parser.add_argument(
        '--no-draw-lines',
        dest='draw_lines',
        action='store_false',
        help='Do not draw connecting lines'
    )

    args = parser.parse_args()

    # Paths
    base_dir = Path(__file__).parent
    features_path = base_dir / "results" / "feature_vis" / "features" / "features.pkl"
    metadata_path = base_dir / "results" / "feature_vis" / "images" / "metadata.json"
    output_dir = base_dir / "results" / "feature_vis" / "plots" / "pairs"

    # Check if features exist
    if not features_path.exists():
        print(f"Error: Features not found at {features_path}")
        print(f"Please run extract_features.py first!")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        sys.exit(1)

    # Create visualizations
    if args.method == "umap":
        visualizer = PairwiseVisualizer(features_path, metadata_path, output_dir)
        visualizer.visualize_all(method="umap", draw_lines=args.draw_lines)
    elif args.method == "tsne":
        visualizer = PairwiseVisualizer(features_path, metadata_path, output_dir)
        visualizer.visualize_all(method="tsne", draw_lines=args.draw_lines)
    else:  # both
        visualizer = PairwiseVisualizer(features_path, metadata_path, output_dir)
        visualizer.visualize_all(method="tsne", draw_lines=args.draw_lines)
        visualizer.visualize_all(method="umap", draw_lines=args.draw_lines)

        print("\n" + "="*60)
        print("✓ All pairwise visualizations complete!")
        print("="*60)
