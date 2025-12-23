#!/bin/bash
# Quick build script for Vello GPU Renderer

set -e  # Exit on error

echo "======================================================================"
echo "Building Vello GPU Renderer for OCRFlow"
echo "======================================================================"
echo ""

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "ERROR: Rust is not installed"
    echo ""
    echo "Install Rust with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    echo "  source \$HOME/.cargo/env"
    exit 1
fi

echo "✓ Rust version: $(rustc --version)"

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "ERROR: maturin is not installed"
    echo ""
    echo "Install maturin with:"
    echo "  pip install maturin"
    exit 1
fi

echo "✓ maturin version: $(maturin --version)"

# Check Vulkan
if ! command -v vulkaninfo &> /dev/null; then
    echo "WARNING: vulkaninfo not found. Installing Vulkan dependencies..."
    sudo apt-get update
    sudo apt-get install -y libvulkan-dev vulkan-tools libxcb1-dev libfontconfig1-dev pkg-config
fi

echo "✓ Vulkan available"
echo ""

# Build
echo "Building Vello renderer (this may take 5-10 minutes on first build)..."
echo ""

if [ "$1" == "--release" ]; then
    echo "Building in RELEASE mode (optimized, slower compile)..."
    maturin develop --release
else
    echo "Building in DEBUG mode (faster compile, slower runtime)..."
    echo "Use './build.sh --release' for production build"
    maturin develop
fi

echo ""
echo "======================================================================"
echo "Build complete!"
echo "======================================================================"
echo ""

# Test import
echo "Testing Python import..."
python3 -c "
import vello_renderer
print(f'✓ vello_renderer v{vello_renderer.__version__} imported successfully')
"

echo ""
echo "Testing VelloRenderer class..."
python3 -c "
from Renderer import VelloRenderer
renderer = VelloRenderer(width=640, height=640)
print(f'✓ VelloRenderer initialized: {renderer}')
"

echo ""
echo "======================================================================"
echo "Success! Vello renderer is ready to use."
echo "======================================================================"
echo ""
echo "Quick start:"
echo "  from Renderer import VelloRenderer"
echo "  renderer = VelloRenderer()"
echo "  images = renderer.render_batch(['Hello', 'World'])"
echo ""
echo "Run benchmarks:"
echo "  python scripts/benchmark_vello.py"
echo ""
