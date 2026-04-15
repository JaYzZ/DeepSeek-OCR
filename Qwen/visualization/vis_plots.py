from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import torch
import umap

from .vis_core import PathManager
from .vis_features import FeatureData, extract_sam_comparison
from .vis_tools import get_pca_map


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300


@dataclass
class PlotConfig:
    """Configuration for plot styling."""
    figsize: Tuple[int, int] = (12, 10)
    dpi: int = 150
    style: str = "whitegrid"
    color_palette: str = "husl"
    title_fontsize: int = 14
    label_fontsize: int = 11
    legend_fontsize: int = 10


class SingleImageVisualizer:
    """Visualize features for a single image across multiple layers."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.path_manager = PathManager(output_dir)
        self.output_dir = self.path_manager.get_output_dir("single_image")

    def plot_feature_comparison(
        self,
        image: Image.Image,
        features_dict: Dict[str, np.ndarray],
        title: str,
        output_name: str,
        layout: str = "horizontal",
    ):
        """
        Plot image with feature visualizations.

        Args:
            image: PIL Image
            features_dict: Dict mapping title to feature array
                For spatial features: [1, H, W, C]
                For pooled features: [hidden_dim]
            title: Overall plot title
            output_name: Output filename
            layout: "horizontal" or "grid"
        """
        # Determine layout
        n_plots = len(features_dict) + 1  # +1 for original image

        if layout == "horizontal":
            fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
            axes = [axes] if n_plots == 1 else axes
        else:  # grid
            ncols = min(4, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
            axes = axes.flatten() if nrows > 1 else [axes]

        # Plot original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')

        # Plot each feature
        for idx, (feat_name, features) in enumerate(features_dict.items(), 1):
            if idx >= len(axes):
                break

            # Convert to appropriate format
            if features.ndim == 4:  # [1, H, W, C] - spatial
                pca_map = get_pca_map(features, img_size=(640, 640))
                axes[idx].imshow(pca_map)
                axes[idx].set_title(feat_name, fontsize=10, fontweight='bold')
            elif features.ndim == 2:  # [H, W] - already 2D
                axes[idx].imshow(features, cmap='viridis')
                axes[idx].set_title(feat_name, fontsize=10, fontweight='bold')
            else:
                # 1D feature - just show as text
                axes[idx].text(
                    0.5, 0.5,
                    f"{feat_name}\nShape: {features.shape}",
                    ha='center', va='center',
                    transform=axes[idx].transAxes
                )
                axes[idx].set_title(feat_name, fontsize=10, fontweight='bold')

            axes[idx].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save
        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_sam_pipeline(
        self,
        image: Image.Image,
        sam_40x40: np.ndarray,  # [1, 40, 40, 768] or [40, 40, 768]
        dpsk_10x10: np.ndarray,  # [100, 1280] or [10, 10, 1280]
        title: str = "SAM Feature Compression Pipeline",
        output_name: str = "sam_pipeline.png",
    ):
        """
        Plot SAM feature compression pipeline.

        Shows: Original | SAM 40x40 | DPSK 10x10
        """
        # Ensure correct shapes
        if sam_40x40.ndim == 4:
            sam_40x40 = sam_40x40[0]  # Remove batch dim
        if sam_40x40.shape[-1] == 768:
            pass  # Already in [H, W, C] format

        # Reshape DPSK to spatial
        if dpsk_10x10.ndim == 1 and dpsk_10x10.shape[0] == 1280:
            # Pooled - can't visualize spatially
            dpsk_10x10 = dpsk_10x10.reshape(1, 1, -1)
        elif dpsk_10x10.ndim == 2 and dpsk_10x10.shape[0] == 100:
            dpsk_10x10 = dpsk_10x10.reshape(1, 10, 10, -1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        axes[0].imshow(image)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')

        # SAM 40x40
        pca_sam = get_pca_map(torch.from_numpy(sam_40x40).unsqueeze(0).float(), img_size=(640, 640))
        axes[1].imshow(pca_sam)
        axes[1].set_title('SAM 40x40x768\n(Before Neck)', fontsize=10, fontweight='bold')
        axes[1].axis('off')

        # DPSK 10x10 - need to reshape to 4D for get_pca_map
        if dpsk_10x10.ndim == 4:  # [1, 10, 10, C]
            dpsk_4d = dpsk_10x10
        elif dpsk_10x10.ndim == 3:
            # Check if it's [1, 1, C] (pooled to 1x1) or [H, W, C]
            if dpsk_10x10.shape[0] == 1 and dpsk_10x10.shape[1] == 1:
                # [1, 1, C] - spatially pooled to 1x1, use placeholder
                dpsk_4d = None
            else:
                # [H, W, C] - reshape to [1, H, W, C]
                h, w = dpsk_10x10.shape[:2]
                dpsk_4d = dpsk_10x10.reshape(1, h, w, -1)
        elif dpsk_10x10.ndim == 2:
            if dpsk_10x10.shape[0] == 100:  # [100, C]
                # Reshape [100, C] to [1, 10, 10, C]
                dpsk_4d = dpsk_10x10.reshape(1, 10, 10, -1)
            else:
                # Unknown 2D format
                dpsk_4d = None
        elif dpsk_10x10.ndim == 1:  # [C] - pooled features
            dpsk_4d = None
        else:
            # Unknown format
            dpsk_4d = None

        if dpsk_4d is not None:
            pca_dpsk = get_pca_map(torch.from_numpy(dpsk_4d).float(), img_size=(640, 640))
        else:
            # For 1x1 or 1D pooled features, create a uniform visualization
            pca_dpsk = np.zeros((640, 640, 3), dtype=np.uint8) + 128  # Gray placeholder
        axes[2].imshow(pca_dpsk)
        axes[2].set_title('DPSK 10x10x1280\n(After Neck)', fontsize=10, fontweight='bold')
        axes[2].axis('off')

        fig.suptitle(title, fontsize=12, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        return output_path


class BatchVisualizer:
    """Visualize features for multiple images."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.path_manager = PathManager(output_dir)
        self.output_dir = self.path_manager.get_output_dir("batch")

    def plot_batch_sam_comparison(
        self,
        images: List[Tuple[Image.Image, str]],  # (image, sample_id)
        sam_features_list: List[np.ndarray],
        dpsk_features_list: List[np.ndarray],
        task_dir: str,
    ):
        """
        Create SAM pipeline visualizations for a batch of images.

        Args:
            images: List of (image, sample_id) tuples
            sam_features_list: List of SAM 40x40 features
            dpsk_features_list: List of DPSK 10x10 features
            task_dir: Task directory name (e.g., "image_caption")
        """
        task_output_dir = self.output_dir / task_dir
        task_output_dir.mkdir(parents=True, exist_ok=True)

        vis = SingleImageVisualizer(task_output_dir)

        output_paths = []
        for (image, sample_id), sam_feat, dpsk_feat in zip(
            images, sam_features_list, dpsk_features_list
        ):
            output_path = vis.plot_sam_pipeline(
                image,
                sam_feat,
                dpsk_feat,
                title=f"{sample_id} | {task_dir}",
                output_name=f"{sample_id}_sam_comparison.png",
            )
            output_paths.append(output_path)

        return output_paths


class EmbeddingVisualizer:
    """Visualize high-dimensional features using t-SNE/UMAP."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.path_manager = PathManager(output_dir)
        self.output_dir = self.path_manager.get_output_dir("embeddings")

    def plot_tsne(
        self,
        features: np.ndarray,  # [N, D]
        labels: np.ndarray,  # [N]
        title: str = "t-SNE Visualization",
        output_name: str = "tsne.png",
        perplexity: int = 30,
    ):
        """Plot t-SNE embedding."""
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Compute t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            verbose=1,
        )
        embedding = tsne.fit_transform(features_scaled)

        # Plot
        fig, ax = plt.subplots(figsize=PlotConfig().figsize)

        # Get unique labels and colors
        unique_labels = sorted(np.unique(labels))
        colors = sns.color_palette(PlotConfig().color_palette, len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                label=str(label),
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

        ax.set_title(title, fontsize=PlotConfig().title_fontsize, fontweight='bold')
        ax.set_xlabel('t-SNE Dimension 1', fontsize=PlotConfig().label_fontsize)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=PlotConfig().label_fontsize)
        ax.legend(fontsize=PlotConfig().legend_fontsize)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=PlotConfig().dpi, bbox_inches='tight')
        plt.close()

        return output_path

    def plot_umap(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        title: str = "UMAP Visualization",
        output_name: str = "umap.png",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ):
        """Plot UMAP embedding."""
        if umap is None:
            print("UMAP not installed. Install with: pip install umap-learn")
            return None

        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Compute UMAP
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric='cosine',
            random_state=42,
        )
        embedding = reducer.fit_transform(features_scaled)

        # Plot (same as t-SNE)
        fig, ax = plt.subplots(figsize=PlotConfig().figsize)

        unique_labels = sorted(np.unique(labels))
        colors = sns.color_palette(PlotConfig().color_palette, len(unique_labels))

        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[color],
                label=str(label),
                alpha=0.7,
                s=50,
                edgecolors='white',
                linewidth=0.5,
            )

        ax.set_title(title, fontsize=PlotConfig().title_fontsize, fontweight='bold')
        ax.set_xlabel('UMAP Dimension 1', fontsize=PlotConfig().label_fontsize)
        ax.set_ylabel('UMAP Dimension 2', fontsize=PlotConfig().label_fontsize)
        ax.legend(fontsize=PlotConfig().legend_fontsize)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / output_name
        plt.savefig(output_path, dpi=PlotConfig().dpi, bbox_inches='tight')
        plt.close()

        return output_path


# Convenience functions

def plot_sam_single_image(
    image_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Quick SAM pipeline visualization for single image."""
    # Load image
    image = Image.open(image_path).convert('RGB')

    # Extract features
    features_dict = extract_sam_comparison(image_path)

    # Create visualization
    vis = SingleImageVisualizer(output_dir)
    output_path = vis.plot_sam_pipeline(
        image,
        features_dict['sam_40x40'].features,
        features_dict['dpsk_final'].features,
        title=f"SAM Pipeline: {image_path.name}",
        output_name=f"{image_path.stem}_sam_pipeline.png",
    )

    return output_path
