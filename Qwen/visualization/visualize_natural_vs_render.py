#!/usr/bin/env python3
"""
Compare natural images vs rendered text for BOTH encoders in ONE plot.

Creates ONE t-SNE plot combining ALL layers from BOTH encoders:
- DeepSeek OCR: Final, Early (L6), Mid (L12), Late (L18)
- Qwen3VL: Final, Early (DS0), Mid (DS1), Late (DS2)

All features are projected to common dimension using PCA first.

Output: ./results/feature_vis/plots/natural_vs_render/
"""

import sys
import json
from pathlib import Path
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


class NaturalVsRenderedVisualizer:
    """Compare natural vs rendered text for both encoders in ONE plot."""

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

        # Build type array from metadata
        self.types = []
        filename_to_meta = {m['filename']: m for m in self.metadata}

        for fname in self.filenames:
            meta = filename_to_meta.get(fname, {})
            self.types.append(meta.get('type', 'unknown'))

        self.types = np.array(self.types)

        print(f"✓ Loaded {len(self.categories)} samples")
        print(f"  Types: {np.unique(self.types, return_counts=True)}")

    def project_to_common_dim(self, features_list, target_dim=64):
        """Project all feature sets to common dimension using PCA."""
        print(f"\n  Projecting {len(features_list)} feature sets to {target_dim}D using PCA...")

        projected_features = []

        # Apply PCA to each feature set individually
        for i, feat_array in enumerate(features_list):
            orig_dim = feat_array.shape[1]

            # If already at target dimension, skip
            if orig_dim == target_dim:
                projected_features.append(feat_array)
                print(f"    Set {i}: {orig_dim}D -> {target_dim}D (no projection needed)")
                continue

            # If smaller than target, pad with zeros (unlikely but handle it)
            if orig_dim < target_dim:
                padded = np.zeros((feat_array.shape[0], target_dim))
                padded[:, :orig_dim] = feat_array
                projected_features.append(padded)
                print(f"    Set {i}: {orig_dim}D -> {target_dim}D (padded)")
            else:
                # Use PCA to reduce dimension
                n_components = min(target_dim, orig_dim)
                pca = PCA(n_components=n_components, random_state=42)
                pca.fit(feat_array)
                projected = pca.transform(feat_array)

                # If we got fewer components than target_dim, pad
                if projected.shape[1] < target_dim:
                    padded = np.zeros((projected.shape[0], target_dim))
                    padded[:, :projected.shape[1]] = projected
                    projected_features.append(padded)
                    print(f"    Set {i}: {orig_dim}D -> {target_dim}D (PCA + pad, var={pca.explained_variance_ratio_.sum():.3f})")
                else:
                    projected_features.append(projected)
                    print(f"    Set {i}: {orig_dim}D -> {target_dim}D (PCA, var={pca.explained_variance_ratio_.sum():.3f})")

        return projected_features

    def compute_embedding(self, features: np.ndarray) -> np.ndarray:
        """Compute t-SNE embedding."""
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42,
            verbose=1,
        )
        return tsne.fit_transform(features_scaled)

    def plot_all_in_one(self):
        """Create ONE plot with all 8 feature sets (4 DPSK + 4 Qwen)."""
        # Define all feature sets
        all_configs = [
            # DeepSeek OCR
            ('DPSK-Final', 'final_visual', self.dpsk_features, '#1f77b4', 'DPSK'),
            ('DPSK-L6', 'layer_0_visual', self.dpsk_features, '#ff7f0e', 'DPSK'),
            ('DPSK-L12', 'layer_1_visual', self.dpsk_features, '#2ca02c', 'DPSK'),
            ('DPSK-L18', 'layer_2_visual', self.dpsk_features, '#d62728', 'DPSK'),
            # Qwen3VL
            ('QWEN-Final', 'visual_output', self.qwen_features, '#9467bd', 'Qwen'),
            ('QWEN-DS0', 'deepstack_0', self.qwen_features, '#8c564b', 'Qwen'),
            ('QWEN-DS1', 'deepstack_1', self.qwen_features, '#e377c2', 'Qwen'),
            ('QWEN-DS2', 'deepstack_2', self.qwen_features, '#17becf', 'Qwen'),
        ]

        # Collect all feature arrays
        all_features_raw = []
        all_labels = []  # which config
        all_types = []   # natural vs rendered
        all_encoders = [] # DPSK or Qwen

        for config_name, feat_key, feat_dict, color, encoder in all_configs:
            if feat_key not in feat_dict:
                print(f"    Warning: {feat_key} not found, skipping...")
                continue

            feat_array = np.array(feat_dict[feat_key])
            n_samples = len(feat_array)

            all_features_raw.append(feat_array)
            all_labels.extend([config_name] * n_samples)
            all_types.extend(self.types.tolist())
            all_encoders.extend([encoder] * n_samples)

            print(f"    {config_name}: shape={feat_array.shape}")

        # Project all to common dimension (64 - limited by sample count)
        all_features = self.project_to_common_dim(all_features_raw, target_dim=64)

        # Stack all features
        combined_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        all_types = np.array(all_types)
        all_encoders = np.array(all_encoders)

        print(f"\n  Computing t-SNE for ALL {len(combined_features)} points...")
        embedding = self.compute_embedding(combined_features)

        # Create single large plot
        fig, ax = plt.subplots(figsize=(16, 12))

        # Plot each feature set
        sample_offset = 0
        for config_name, feat_key, feat_dict, color, encoder in all_configs:
            if feat_key not in feat_dict:
                continue

            n_samples = len(feat_dict[feat_key])
            layer_embedding = embedding[sample_offset:sample_offset + n_samples]
            layer_types = all_types[sample_offset:sample_offset + n_samples]

            # Natural images (circles)
            natural_mask = layer_types == 'natural'
            ax.scatter(
                layer_embedding[natural_mask, 0],
                layer_embedding[natural_mask, 1],
                c=color,
                marker='o',
                label=f'{config_name} (Natural)',
                alpha=0.6,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

            # Rendered text (X markers)
            rendered_mask = layer_types == 'rendered_text'
            ax.scatter(
                layer_embedding[rendered_mask, 0],
                layer_embedding[rendered_mask, 1],
                c=color,
                marker='X',
                label=f'{config_name} (Rendered)',
                alpha=0.6,
                s=60,
                edgecolors='white',
                linewidth=0.5,
            )

            sample_offset += n_samples

        # Title and labels
        ax.set_title(
            'All Features: DeepSeek OCR vs Qwen3VL - Natural vs Rendered Text',
            fontsize=14, fontweight='bold', pad=15
        )
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.2)

        # Create legend (2 columns: by feature set)
        ax.legend(
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            frameon=True,
            fontsize=8,
            ncol=1,
        )

        plt.tight_layout()

        # Save
        output_path = self.output_dir / "all_features_combined.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ Saved: {output_path}")
        plt.close()

        # Compute and print pair distance statistics
        print(f"\n  Pair Distance Statistics (t-SNE space):")
        print(f"    {'Feature Set':<15} {'μ':<8} {'σ':<8}")
        print(f"    {'-'*35}")

        sample_offset = 0
        for config_name, feat_key, feat_dict, color, encoder in all_configs:
            if feat_key not in feat_dict:
                continue

            n_samples = len(feat_dict[feat_key])
            layer_embedding = embedding[sample_offset:sample_offset + n_samples]

            # For each position in this layer, compute pair distance
            distances = []
            for i in range(n_samples):
                if self.types[i] == 'natural':
                    natural_fname = self.filenames[i]
                    render_fname = natural_fname.replace('.jpg', '_render.png')

                    # Find rendered version index in global filenames array
                    render_idx_array = np.where(self.filenames == render_fname)[0]
                    if len(render_idx_array) > 0:
                        j = render_idx_array[0]
                        # j is in the same relative position in this layer
                        if j < n_samples and self.types[j] == 'rendered_text':
                            dist = np.linalg.norm(layer_embedding[i] - layer_embedding[j])
                            distances.append(dist)

            if distances:
                print(f"    {config_name:<15} {np.mean(distances):<8.3f} {np.std(distances):<8.3f}")
            else:
                print(f"    {config_name:<15} No pairs found")

            sample_offset += n_samples

    def plot_distance_comparison(self):
        """Create side-by-side comparison of distance distributions."""
        dpsk_layers = [
            ('Final', 'final_visual', self.dpsk_features),
            ('L6', 'layer_0_visual', self.dpsk_features),
            ('L12', 'layer_1_visual', self.dpsk_features),
            ('L18', 'layer_2_visual', self.dpsk_features),
        ]

        qwen_layers = [
            ('Final', 'visual_output', self.qwen_features),
            ('DS0', 'deepstack_0', self.qwen_features),
            ('DS1', 'deepstack_1', self.qwen_features),
            ('DS2', 'deepstack_2', self.qwen_features),
        ]

        # Collect distance data
        dpsk_data = []
        qwen_data = []
        layer_names = []

        for (name_d, key_d, _), (name_q, key_q, _) in zip(dpsk_layers, qwen_layers):
            layer_names.append(name_d)

            # Compute distances for DPSK
            if key_d in self.dpsk_features:
                feat_array = np.array(self.dpsk_features[key_d])
                distances = []
                for i in range(len(self.types)):
                    if self.types[i] == 'natural':
                        natural_fname = self.filenames[i]
                        render_fname = natural_fname.replace('.jpg', '_render.png')
                        j = np.where(self.filenames == render_fname)[0]
                        if len(j) > 0:
                            dist = np.linalg.norm(feat_array[i] - feat_array[j[0]])
                            distances.append(dist)
                if distances:
                    dpsk_data.append(distances)

            # Compute distances for Qwen
            if key_q in self.qwen_features:
                feat_array = np.array(self.qwen_features[key_q])
                distances = []
                for i in range(len(self.types)):
                    if self.types[i] == 'natural':
                        natural_fname = self.filenames[i]
                        render_fname = natural_fname.replace('.jpg', '_render.png')
                        j = np.where(self.filenames == render_fname)[0]
                        if len(j) > 0:
                            dist = np.linalg.norm(feat_array[i] - feat_array[j[0]])
                            distances.append(dist)
                if distances:
                    qwen_data.append(distances)

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # DeepSeek OCR
        bp1 = axes[0].boxplot(dpsk_data, tick_labels=layer_names, patch_artist=True, showmeans=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('#1f77b4')
            patch.set_alpha(0.6)

        axes[0].set_title('DeepSeek OCR', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Pair Distance (Original Feature Space)', fontsize=10)
        axes[0].grid(True, alpha=0.3, axis='y')

        for i, dists in enumerate(dpsk_data):
            mean_val = np.mean(dists)
            axes[0].text(i + 1, mean_val, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)

        # Qwen3VL
        bp2 = axes[1].boxplot(qwen_data, tick_labels=layer_names, patch_artist=True, showmeans=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('#ff7f0e')
            patch.set_alpha(0.6)

        axes[1].set_title('Qwen3VL', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Pair Distance (Original Feature Space)', fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')

        for i, dists in enumerate(qwen_data):
            mean_val = np.mean(dists)
            axes[1].text(i + 1, mean_val, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle(
            'Pair Distance Comparison: Natural Image vs Rendered Text',
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        # Save
        output_path = self.output_dir / "distance_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path}")
        plt.close()

    def visualize_all(self):
        """Generate all natural vs rendered comparison visualizations."""
        print(f"\n{'='*60}")
        print(f"Generating Natural vs Rendered Text Comparisons")
        print(f"{'='*60}\n")

        print("Creating combined plot with ALL features (8 sets total)...")
        self.plot_all_in_one()

        print("\nCreating distance comparison...")
        self.plot_distance_comparison()

        print(f"\n{'='*60}")
        print(f"✓ All visualizations saved to {self.output_dir}/")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    features_path = base_dir / "results" / "feature_vis" / "features" / "features.pkl"
    metadata_path = base_dir / "results" / "feature_vis" / "images" / "metadata.json"
    output_dir = base_dir / "results" / "feature_vis" / "plots" / "natural_vs_render"

    # Check if features exist
    if not features_path.exists():
        print(f"Error: Features not found at {features_path}")
        print(f"Please run extract_features.py first!")
        sys.exit(1)

    if not metadata_path.exists():
        print(f"Error: Metadata not found at {metadata_path}")
        sys.exit(1)

    # Create visualizations
    visualizer = NaturalVsRenderedVisualizer(features_path, metadata_path, output_dir)
    visualizer.visualize_all()
