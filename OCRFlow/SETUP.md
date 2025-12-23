# Vello Renderer Setup Guide

Complete step-by-step guide to build and install the GPU-accelerated Vello renderer for OCRFlow.

## Overview

The Vello renderer provides:
- **GPU-accelerated rendering**: Uses Vulkan compute shaders for 2D graphics
- **12x faster than PIL**: 1,565 img/s vs 130 img/s
- **CJK support**: Chinese, Japanese, Korean via Noto Sans fonts
- **CPU-based**: Frees up GPU resources for training

**Technology stack:**
- **Vello**: GPU compute shader-based 2D rendering (Rust)
- **cosmic-text**: Advanced text layout and shaping (Rust)
- **PyO3**: Rust-Python bindings
- **maturin**: Build tool for Rust Python extensions

---

## Step 1: Install Rust Toolchain

### 1.1 Install Rust via rustup

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Add to PATH (or restart shell)
source $HOME/.cargo/env

# Verify installation
rustc --version
cargo --version
```

**Expected output:**
```
rustc 1.75.0 (or newer)
cargo 1.75.0 (or newer)
```

---

## Step 2: Install System Dependencies

### 2.1 Install Vulkan Libraries

Vello uses Vulkan for GPU compute rendering:

```bash
# Update package list
sudo apt-get update

# Install Vulkan development libraries
sudo apt-get install -y \
    libvulkan-dev \
    vulkan-tools \
    libxcb1-dev \
    libfontconfig1-dev \
    pkg-config
```

### 2.2 Verify Vulkan Support

```bash
# Check Vulkan devices
vulkaninfo | grep deviceName
```

**Expected output:**
```
deviceName = NVIDIA GeForce RTX 4090
(or similar GPU name)
```

**If no devices found:**
```bash
# For NVIDIA GPUs, install Vulkan drivers
sudo apt-get install -y nvidia-vulkan-driver

# Verify NVIDIA driver
nvidia-smi
```

---

## Step 3: Install CJK Fonts

### 3.1 Install Noto Sans CJK

Required for Chinese, Japanese, Korean character rendering:

```bash
# Install Noto Sans CJK fonts
sudo apt-get install -y fonts-noto-cjk

# Verify installation
ls /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

**Expected output:**
```
/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc
```

---

## Step 4: Install maturin

maturin is the build tool for Rust Python extensions:

```bash
# Install maturin
pip install maturin

# Verify installation
maturin --version
```

**Expected output:**
```
maturin 1.4.0 (or newer)
```

---

## Step 5: Build Vello Renderer

### 5.1 Navigate to Vello Renderer Directory

```bash
cd /share/project/xiyan/sources/DeepSeek-OCR/Renderer
```

### 5.2 Build and Install

```bash
# Build and install in development mode (recommended)
maturin develop --release

# Alternative: Build wheel and install manually
# maturin build --release
# pip install target/wheels/vello_renderer-*.whl
```

**Expected output:**
```
🔗 Found pyo3 bindings
🐍 Found CPython 3.12
📦 Built wheel to target/wheels/vello_renderer-0.1.0-cp312-cp312-linux_x86_64.whl
✨ Successfully installed vello_renderer-0.1.0
```

**Build time:**
- First build: 3-5 minutes (compiles Vello, wgpu, cosmic-text)
- Subsequent builds: 30 seconds (incremental compilation)

---

## Step 6: Verify Installation

### 6.1 Test Basic Import

```bash
# Test import
python -c "import vello_renderer; print('✓ Vello renderer installed successfully')"
```

### 6.2 Test Rendering Functionality

```bash
python -c "
from Renderer import VelloRenderer

renderer = VelloRenderer(width=640, height=640)
images = renderer.render_batch(['Hello Vello!', 'GPU Rendering'])
print(f'✓ Rendered {len(images)} images')
print(f'✓ Image shape: {images[0].shape}')
print(f'✓ Image dtype: {images[0].dtype}')
"
```

**Expected output:**
```
✓ Rendered 2 images
✓ Image shape: (640, 640, 3)
✓ Image dtype: uint8
```

### 6.3 Test CJK Support

```bash
python -c "
from Renderer import VelloRenderer

renderer = VelloRenderer()
texts = [
    'Hello World',
    '你好世界',  # Chinese
    'こんにちは',  # Japanese
    '안녕하세요'  # Korean
]
images = renderer.render_batch(texts)
print(f'✓ Rendered {len(images)} multilingual images')
"
```

**Expected output:**
```
✓ Rendered 4 multilingual images
```

---

## Step 7: Run Performance Benchmark

### 7.1 Benchmark All Renderers

```bash
cd /share/project/xiyan/sources/DeepSeek-OCR/OCRFlow
python scripts/benchmark_renderers.py
```

**Expected output:**
```
Renderer Benchmarks
======================================================================
Testing with 1000 samples...

Vello Renderer: 1,565 img/s (12.0x faster than PIL)
Skia Renderer: 778 img/s (6.0x faster than PIL)
PIL Renderer: 130 img/s (baseline)
```

### 7.2 Visual Inspection

Create test renders to verify quality:

```bash
python -c "
from Renderer import VelloRenderer
from PIL import Image

renderer = VelloRenderer()
texts = [
    'The quick brown fox jumps over the lazy dog.',
    '人工智能技术的发展日新月异。深度学习模型在自然语言处理、计算机视觉等领域取得了突破性进展。',
    'Mixed: Hello 世界 こんにちは 안녕하세요!'
]

images = renderer.render_batch_pil(texts)
for i, img in enumerate(images):
    img.save(f'/tmp/vello_test_{i}.png')
print('✓ Saved test images to /tmp/vello_test_*.png')
"
```

Inspect the images to verify:
- Text is clear and readable
- CJK characters render correctly
- Multi-line text wraps properly
- Font size is appropriate

---

## Step 8: Integration with OCRFlow Training

### 8.1 Verify Integration

The Vello renderer is automatically used in training if installed:

```bash
# Check if Vello is detected
python -c "
from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder
encoder = DPSKOCREncoder(device='cuda:0')
print('✓ Vision encoder created with Vello renderer')
"
```

### 8.2 Run Test Training

```bash
# Run short training test
python examples/train.py --max_steps 10
```

**Check logs for Vello usage:**
```bash
grep "Vello" logs/training_*.log
```

**Expected output:**
```
Using Vello renderer (1565 img/s)
```

---

## Troubleshooting

### Issue 1: Rust Installation Failed

**Symptom:**
```
curl: command not found
```

**Solution:**
```bash
# Install curl first
sudo apt-get install -y curl

# Then retry Rust installation
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
```

---

### Issue 2: Vulkan Not Found

**Symptom:**
```
ERROR: Failed to initialize Vello GPU renderer
```

**Solution 1: Check Vulkan support**
```bash
vulkaninfo | grep deviceName
```

If empty, install Vulkan drivers:
```bash
sudo apt-get install -y vulkan-tools libvulkan1

# For NVIDIA GPUs
sudo apt-get install -y nvidia-vulkan-driver

# Verify driver
nvidia-smi
```

**Solution 2: Check GPU visibility**
```bash
# Verify GPU is accessible
lspci | grep -i vga
```

---

### Issue 3: maturin Build Failed

**Symptom:**
```
error: linker `cc` not found
```

**Solution:**
```bash
# Install build tools
sudo apt-get install -y build-essential

# Retry build
cd utils/vello_renderer
maturin develop --release
```

---

### Issue 4: CJK Characters Show as Boxes

**Symptom:**
Rendered images show boxes (□) instead of Chinese/Japanese/Korean characters

**Solution:**
```bash
# Install Noto Sans CJK fonts
sudo apt-get install -y fonts-noto-cjk

# Rebuild Vello renderer to pick up new fonts
cd utils/vello_renderer
maturin develop --release

# Verify fonts are found
fc-list | grep -i noto | grep -i cjk
```

---

### Issue 5: Import Error After Build

**Symptom:**
```
ImportError: vello_renderer not found
```

**Solution:**
```bash
# Check if wheel was installed
pip list | grep vello

# If not found, reinstall
cd utils/vello_renderer
pip uninstall vello_renderer -y
maturin develop --release
```

---

### Issue 6: Slow Rendering Performance

**Symptom:**
Vello renderer is slower than expected (<1000 img/s)

**Solution 1: Verify GPU is being used**
```bash
# Run benchmark with detailed output
python scripts/benchmark_renderers.py

# Check if Vulkan is using GPU
vulkaninfo | grep deviceType
# Should show: deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
```

**Solution 2: Update GPU drivers**
```bash
# For NVIDIA GPUs
sudo apt-get update
sudo apt-get upgrade nvidia-driver-*

# Reboot
sudo reboot
```

**Solution 3: Clear cache and rebuild**
```bash
cd utils/vello_renderer
cargo clean
maturin develop --release
```

---

### Issue 7: Build Taking Too Long

**Symptom:**
maturin build hangs or takes > 10 minutes

**Solution:**
```bash
# Check system resources
free -h  # Check available RAM (needs ~8GB for build)
df -h    # Check disk space (needs ~5GB)

# If low on resources, build without LTO optimization
cd utils/vello_renderer

# Edit Cargo.toml and remove LTO settings:
# [profile.release]
# opt-level = 3
# # lto = true  # Comment this out
# # codegen-units = 1  # Comment this out

# Rebuild
maturin develop --release
```

---

## Performance Tuning

### Optimal Configuration

For maximum throughput:

```python
from Renderer import VelloRenderer

renderer = VelloRenderer(
    width=640,
    height=640,
    padding=20,
    min_font_size=9,
    max_font_size=20,
)

# Render in batches (GPU efficiency)
batch_size = 64
images = renderer.render_batch(texts[:batch_size])
```

### Memory Usage

Vello renderer memory usage:
- **Initialization**: ~200MB (Vulkan context + fonts)
- **Per image**: ~1.2MB (640x640x3)
- **Batch of 64**: ~200MB + 64×1.2MB = ~280MB

Total system requirements:
- **RAM**: 2GB minimum for Vello
- **GPU VRAM**: Not used (Vello uses CPU-side Vulkan compute)

---

## Summary

**Complete installation (copy-paste):**

```bash
# 1. Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

# 2. Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    libvulkan-dev \
    vulkan-tools \
    libxcb1-dev \
    libfontconfig1-dev \
    fonts-noto-cjk

# 3. Install maturin
pip install maturin

# 4. Build Vello renderer
cd utils/vello_renderer
maturin develop --release

# 5. Test installation
python -c "from Renderer import VelloRenderer; print('✓ Success!')"

# 6. Benchmark
cd ../..
python scripts/benchmark_renderers.py
```

**Expected outcome:**
- ✅ Vello renderer installed and working
- ✅ 1,565 img/s rendering speed (12x faster than PIL)
- ✅ CJK character support verified
- ✅ Ready for OCRFlow training

---

## Frequently Asked Questions

### Q: Do I need a GPU for Vello?

**A:** Vello uses Vulkan compute shaders which can run on CPU or GPU. In most systems, it will automatically use the GPU for compute operations, but the actual rendering work is CPU-bound. You don't need a high-end GPU - even integrated graphics work fine.

### Q: Can I use Vello in Docker?

**A:** Yes, but the Docker container needs:
1. Vulkan libraries installed
2. GPU drivers mounted (if using GPU)
3. Access to `/dev/dri` for Vulkan

Example Dockerfile additions:
```dockerfile
RUN apt-get install -y libvulkan-dev vulkan-tools fonts-noto-cjk
```

### Q: How much does Vello improve training speed?

**A:** With 7 encoder workers, Vello eliminates the rendering bottleneck:
- **Without Vello**: PIL at 130 img/s → Training bottlenecked by rendering
- **With Vello**: 1,565 img/s → Training runs at full encoder speed (2,079 pairs/s)

### Q: Can I use Vello with fewer GPUs?

**A:** Yes! Vello benefits any configuration:
- **1 GPU**: Frees up CPU for other tasks
- **2-4 GPUs**: Ensures rendering doesn't bottleneck encoding
- **8 GPUs**: Full speed encoding without CPU saturation

### Q: Does Vello work on Windows or macOS?

**A:** Theoretically yes (Vulkan is cross-platform), but we only test on Linux. You may need to adjust paths and dependencies.

---

## References

- **Vello**: https://github.com/linebender/vello
- **cosmic-text**: https://github.com/pop-os/cosmic-text
- **PyO3**: https://pyo3.rs
- **maturin**: https://www.maturin.rs
- **Vulkan**: https://www.vulkan.org

---

**Version:** 1.0.0
**Last Updated:** 2025-12-05
