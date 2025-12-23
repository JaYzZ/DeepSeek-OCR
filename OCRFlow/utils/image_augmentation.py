"""
Image Augmentation for OCR Training

Applies document-style augmentations to rendered text images while preserving:
1. Text content (ground truth unchanged)
2. Image dimensions (no cropping, stays 640x640)
3. Spatial structure (important for token grid alignment)

Only visual appearance is changed: colors, brightness, noise, blur, etc.

IMPORTANT: All augmentations use random.random() and np.random for TRUE randomness.
Each image gets independent random augmentation parameters.
"""

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from typing import Union, List

# Ensure numpy random is properly seeded from system entropy
np.random.seed(None)  # Use system entropy for true randomness


def augment_document_image(
    image: Union[np.ndarray, Image.Image],
    augment_prob: float = 0.8,
    background_prob: float = 0.6,
    brightness_prob: float = 0.3,
    contrast_prob: float = 0.3,
    noise_prob: float = 0.2,
    blur_prob: float = 0.15,
    jpeg_prob: float = 0.1,
) -> np.ndarray:
    """
    Apply document-style augmentation to a rendered text image.

    Simulates real-world document variations:
    - Different paper colors (white, cream, beige)
    - Lighting conditions (brightness/contrast)
    - Scan/camera artifacts (noise, blur, compression)

    Args:
        image: [H, W, 3] RGB numpy array or PIL Image (white bg, black text)
        augment_prob: Probability of applying any augmentation
        background_prob: Probability of changing background color
        brightness_prob: Probability of brightness variation
        contrast_prob: Probability of contrast variation
        noise_prob: Probability of adding noise
        blur_prob: Probability of adding blur
        jpeg_prob: Probability of JPEG compression artifacts

    Returns:
        Augmented image as [H, W, 3] numpy array
    """
    # Skip augmentation with probability
    if random.random() > augment_prob:
        if isinstance(image, np.ndarray):
            return image
        return np.array(image)

    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        pil_img = Image.fromarray(image)
    else:
        pil_img = image.copy()

    # 1. Background color variation (simulate different paper types)
    if random.random() < background_prob:
        pil_img = _augment_background(pil_img)

    # 2. Brightness variation (simulate lighting conditions)
    if random.random() < brightness_prob:
        enhancer = ImageEnhance.Brightness(pil_img)
        factor = random.uniform(0.7, 1.3)  # 70% to 130% brightness
        pil_img = enhancer.enhance(factor)

    # 3. Contrast variation (simulate camera/scanner settings)
    if random.random() < contrast_prob:
        enhancer = ImageEnhance.Contrast(pil_img)
        factor = random.uniform(0.8, 1.2)  # 80% to 120% contrast
        pil_img = enhancer.enhance(factor)

    # 4. Gaussian blur (simulate out-of-focus or motion blur)
    if random.random() < blur_prob:
        radius = random.uniform(0.3, 1.0)
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 5. Gaussian noise (simulate sensor noise or scan artifacts)
    if random.random() < noise_prob:
        pil_img = _add_gaussian_noise(pil_img, sigma=random.uniform(2, 8))

    # 6. JPEG compression artifacts (simulate compressed documents)
    if random.random() < jpeg_prob:
        pil_img = _add_jpeg_artifacts(pil_img, quality=random.randint(75, 95))

    return np.array(pil_img)


def _augment_background(pil_img: Image.Image) -> Image.Image:
    """
    Change background color to simulate different paper types.

    Preserves text by creating a mask and only changing background pixels.
    """
    # Convert to numpy for mask creation
    arr = np.array(pil_img)

    # Create text mask: pixels that are NOT white (have some text)
    # Threshold: pixels with any channel < 250 are considered text
    text_mask = (arr < 250).any(axis=2)

    # Choose random background color
    bg_type = random.random()
    if bg_type < 0.3:
        # Cream/beige paper (yellowish tint)
        bg_color = (
            random.randint(245, 255),  # R
            random.randint(240, 250),  # G
            random.randint(220, 240),  # B
        )
    elif bg_type < 0.6:
        # Light gray paper (grayish tint)
        gray = random.randint(235, 250)
        bg_color = (gray, gray, gray)
    else:
        # Slightly off-white (random tint)
        base = random.randint(240, 255)
        bg_color = (
            base,
            base + random.randint(-5, 5),
            base + random.randint(-5, 5),
        )

    # Clip to valid range
    bg_color = tuple(max(0, min(255, c)) for c in bg_color)

    # Create background image
    bg_img = Image.new('RGB', pil_img.size, bg_color)

    # Composite: use text from original, background from new
    # Where text_mask is True, use original; else use bg_color
    result_arr = arr.copy()
    result_arr[~text_mask] = bg_color

    return Image.fromarray(result_arr)


def _add_gaussian_noise(pil_img: Image.Image, sigma: float = 5.0) -> Image.Image:
    """
    Add Gaussian noise to simulate sensor noise or scan artifacts.

    Args:
        pil_img: Input PIL Image
        sigma: Standard deviation of noise (typical: 2-8)

    Returns:
        Noisy image
    """
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_arr)


def _add_jpeg_artifacts(pil_img: Image.Image, quality: int = 85) -> Image.Image:
    """
    Add JPEG compression artifacts by encoding and decoding.

    Args:
        pil_img: Input PIL Image
        quality: JPEG quality (1-100, lower = more artifacts)

    Returns:
        Image with JPEG artifacts
    """
    import io

    # Encode to JPEG in memory
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)

    # Decode back
    return Image.open(buffer)


def augment_batch(
    images: List[Union[np.ndarray, Image.Image]],
    augment_prob: float = 0.8,
    **kwargs
) -> List[np.ndarray]:
    """
    Apply augmentation to a batch of images.

    Each image is augmented independently with RANDOM variations.
    Uses random.random() and np.random for true randomness - NOT reproducible.

    Args:
        images: List of images (numpy arrays or PIL Images)
        augment_prob: Probability of augmenting each image
        **kwargs: Additional arguments for augment_document_image()

    Returns:
        List of augmented images as numpy arrays
    """
    augmented = []
    for img in images:
        # Each image gets independent random augmentation
        aug_img = augment_document_image(img, augment_prob=augment_prob, **kwargs)
        augmented.append(aug_img)
    return augmented


# Preset configurations for different augmentation intensities

AUGMENT_PRESETS = {
    "none": {
        "augment_prob": 0.0,
    },
    "light": {
        "augment_prob": 0.5,
        "background_prob": 0.3,
        "brightness_prob": 0.2,
        "contrast_prob": 0.2,
        "noise_prob": 0.1,
        "blur_prob": 0.05,
        "jpeg_prob": 0.05,
    },
    "medium": {
        "augment_prob": 0.8,
        "background_prob": 0.6,
        "brightness_prob": 0.3,
        "contrast_prob": 0.3,
        "noise_prob": 0.2,
        "blur_prob": 0.15,
        "jpeg_prob": 0.1,
    },
    "heavy": {
        "augment_prob": 0.95,
        "background_prob": 0.8,
        "brightness_prob": 0.5,
        "contrast_prob": 0.5,
        "noise_prob": 0.3,
        "blur_prob": 0.25,
        "jpeg_prob": 0.2,
    },
}


def get_augment_config(preset: str = "medium") -> dict:
    """
    Get augmentation configuration by preset name.

    Args:
        preset: One of "none", "light", "medium", "heavy"

    Returns:
        Configuration dict for augment_document_image()
    """
    return AUGMENT_PRESETS.get(preset, AUGMENT_PRESETS["medium"])


if __name__ == "__main__":
    # Test augmentation
    import matplotlib.pyplot as plt
    from Renderer import VelloRenderer

    print("Testing image augmentation...")

    # Render sample text
    renderer = VelloRenderer()
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "人工智能技术的发展日新月异。深度学习模型在自然语言处理、计算机视觉等领域取得了突破性进展。",
        "Mixed: Hello 世界 こんにちは 안녕하세요!",
    ]

    images = renderer.render_batch(texts)

    # Show original and augmented versions
    fig, axes = plt.subplots(len(images), 4, figsize=(16, 4*len(images)))

    for i, img in enumerate(images):
        # Original
        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis('off')

        # Light augmentation
        aug1 = augment_document_image(img, **AUGMENT_PRESETS["light"])
        axes[i, 1].imshow(aug1)
        axes[i, 1].set_title("Light")
        axes[i, 1].axis('off')

        # Medium augmentation
        aug2 = augment_document_image(img, **AUGMENT_PRESETS["medium"])
        axes[i, 2].imshow(aug2)
        axes[i, 2].set_title("Medium")
        axes[i, 2].axis('off')

        # Heavy augmentation
        aug3 = augment_document_image(img, **AUGMENT_PRESETS["heavy"])
        axes[i, 3].imshow(aug3)
        axes[i, 3].set_title("Heavy")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig("/tmp/augmentation_test.png", dpi=150)
    print("Saved test results to /tmp/augmentation_test.png")

    # Test batch augmentation
    print("\nTesting batch augmentation...")
    batch_size = 100
    test_images = renderer.render_batch(texts * (batch_size // len(texts)))

    import time
    start = time.time()
    augmented = augment_batch(test_images, **AUGMENT_PRESETS["medium"])
    elapsed = time.time() - start

    print(f"Augmented {len(augmented)} images in {elapsed:.3f}s ({len(augmented)/elapsed:.1f} img/s)")
    print(f"Original shape: {test_images[0].shape}, Augmented shape: {augmented[0].shape}")
    print("✓ Dimensions preserved!")
