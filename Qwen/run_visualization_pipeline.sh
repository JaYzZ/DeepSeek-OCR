#!/bin/bash
# Complete pipeline for t-SNE feature visualization
# Run from the Qwen/ directory

set -e  # Exit on error

# Parse command line arguments
USE_CLS=true
VISUALIZE_CLS=true
VISUALIZE_VISUAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --visualize-visual)
            VISUALIZE_CLS=false
            VISUALIZE_VISUAL=true
            shift
            ;;
        --visualize-all)
            VISUALIZE_CLS=true
            VISUALIZE_VISUAL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--visualize-visual] [--visualize-all]"
            exit 1
            ;;
    esac
done

echo "======================================================================"
echo "t-SNE Feature Visualization Pipeline"
echo "======================================================================"
echo ""
echo "This pipeline will:"
echo "  1. Generate diverse image dataset (5 categories × 50 samples)"
echo "  2. Extract features from DeepSeek OCR and Qwen3VL"
echo "     - CLS tokens + visual tokens (attention-weighted pooling)"
echo "  3. Compute t-SNE embeddings and create visualizations"
echo ""
echo "Visualization options:"
echo "  - CLS tokens: $VISUALIZE_CLS"
echo "  - Visual tokens: $VISUALIZE_VISUAL"
echo ""
echo "Output: ./results/feature_vis/"
echo "======================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Generate dataset
echo "Step 1/3: Generating visualization dataset..."
echo "----------------------------------------------------------------------"
python generate_visualization_data.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to generate dataset"
    exit 1
fi
echo ""

# Step 2: Extract features
echo "Step 2/3: Extracting features from vision encoders..."
echo "----------------------------------------------------------------------"
python extract_features.py
if [ $? -ne 0 ]; then
    echo "Error: Failed to extract features"
    exit 1
fi
echo ""

# Step 3: Create visualizations
echo "Step 3/3: Creating t-SNE visualizations..."
echo "----------------------------------------------------------------------"

# Create plots directory
mkdir -p results/feature_vis/plots_cls
mkdir -p results/feature_vis/plots_visual

if [ "$VISUALIZE_CLS" = true ]; then
    echo ""
    echo "Creating CLS token visualizations..."
    python visualize_tsne.py --use-cls
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create CLS visualizations"
        exit 1
    fi
    # Move CLS plots to subdirectory
    mv results/feature_vis/plots/*.png results/feature_vis/plots_cls/ 2>/dev/null || true
fi

if [ "$VISUALIZE_VISUAL" = true ]; then
    echo ""
    echo "Creating visual token visualizations..."
    python visualize_tsne.py --no-use-cls
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create visual token visualizations"
        exit 1
    fi
    # Move visual plots to subdirectory
    mv results/feature_vis/plots/*.png results/feature_vis/plots_visual/ 2>/dev/null || true
fi

echo ""

# Summary
echo "======================================================================"
echo "✓ Pipeline Complete!"
echo "======================================================================"
echo ""
echo "Results saved to: ./results/feature_vis/"
echo ""
echo "Directory structure:"
echo "  ./results/feature_vis/"
echo "    ├── images/          # Generated images (250 samples)"
echo "    ├── features/        # Extracted features (.pkl)"
if [ "$VISUALIZE_CLS" = true ]; then
    echo "    ├── plots_cls/      # CLS token t-SNE visualizations (.png)"
fi
if [ "$VISUALIZE_VISUAL" = true ]; then
    echo "    ├── plots_visual/   # Visual token t-SNE visualizations (.png)"
fi
echo ""
echo "Feature structure:"
echo "  DeepSeek OCR:"
echo "    - layer_0_cls: CLS token from CLIP layer 6"
echo "    - layer_0_visual: Pooled visual tokens from CLIP layer 6"
echo "    - layer_1_cls: CLS token from CLIP layer 12"
echo "    - layer_1_visual: Pooled visual tokens from CLIP layer 12"
echo "    - layer_2_cls: CLS token from CLIP layer 18"
echo "    - layer_2_visual: Pooled visual tokens from CLIP layer 18"
echo "    - final_visual: Pooled final visual tokens (no CLS)"
echo ""
echo "Key visualizations:"
if [ "$VISUALIZE_CLS" = true ]; then
    echo "  CLS tokens (plots_cls/):"
    echo "    - dpsk_layer_*.png         # DeepSeek CLS at different layers"
    echo "    - all_layers_comparison.png # Grid view of all CLS layers"
fi
if [ "$VISUALIZE_VISUAL" = true ]; then
    echo "  Visual tokens (plots_visual/):"
    echo "    - dpsk_layer_*.png         # DeepSeek visual tokens (attention-weighted)"
    echo "    - all_layers_comparison.png # Grid view of all visual layers"
fi
echo ""
echo "======================================================================"
