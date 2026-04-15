#!/usr/bin/env python3
"""Generate diverse image dataset for t-SNE visualization.

This script creates a balanced dataset covering:
1. Pure text (various fonts, densities, languages)
2. Natural images (objects, scenes)
3. Documents (mixed text/graphics)
4. Simple to complex visual patterns

Output: ./results/feature_vis/images/ with categorized samples
"""

import os
import random
from pathlib import Path

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from Renderer import VelloRenderer

HAS_VELLO = True
print("Using Vello renderer for text generation")


class VisualizationDataGenerator:
    """Generate diverse images for feature visualization."""

    def __init__(self, output_dir: str = "./results/feature_vis/images", samples_per_category: int = 50):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.samples_per_category = samples_per_category
        self.image_size = (640, 640)
        self.metadata = []

        # Initialize renderer
        if HAS_VELLO:
            self.renderer = VelloRenderer()
        else:
            self.renderer = None

    def generate_all(self):
        """Generate all categories of images."""
        print(f"\nGenerating {self.samples_per_category} samples per category...")

        # 1. Pure text - various densities and styles
        self._generate_text_samples()

        # 2. Synthetic patterns and shapes
        self._generate_pattern_samples()

        # 3. Mixed text and graphics
        self._generate_mixed_samples()

        # 4. Gradients and color variations
        self._generate_gradient_samples()

        # 5. Different text languages and scripts
        self._generate_multilingual_samples()

        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n✓ Generated {len(self.metadata)} images in {self.output_dir}")
        print(f"✓ Metadata saved to {metadata_path}")

        return self.metadata

    def _save_image(self, img: Image.Image, category: str, idx: int, **meta):
        """Save image and record metadata."""
        filename = f"{category}_{idx:04d}.png"
        filepath = self.output_dir / filename
        img.save(filepath)

        self.metadata.append({
            "filename": filename,
            "category": category,
            "index": idx,
            **meta
        })

    def _generate_text_samples(self):
        """Generate pure text images with varying density and style."""
        category = "text"
        print(f"\nGenerating {category} samples...")

        # Text samples with different characteristics
        text_configs = [
            # Sparse text
            {"words": 50, "font_size": 24, "line_spacing": 1.5, "density": "sparse"},
            {"words": 80, "font_size": 20, "line_spacing": 1.4, "density": "sparse"},
            # Medium density
            {"words": 150, "font_size": 18, "line_spacing": 1.3, "density": "medium"},
            {"words": 200, "font_size": 16, "line_spacing": 1.2, "density": "medium"},
            # Dense text
            {"words": 300, "font_size": 14, "line_spacing": 1.1, "density": "dense"},
            {"words": 400, "font_size": 12, "line_spacing": 1.0, "density": "dense"},
        ]

        for idx in range(self.samples_per_category):
            config = random.choice(text_configs)
            text = self._generate_lorem_ipsum(config["words"])

            if HAS_VELLO:
                img_array = self.renderer.render(text, width=640, height=640)
                img = Image.fromarray(img_array)
            else:
                img = self._render_text_pil(text, config["font_size"], config["line_spacing"])

            self._save_image(img, category, idx, **config)

    def _generate_pattern_samples(self):
        """Generate geometric patterns and shapes."""
        category = "pattern"
        print(f"\nGenerating {category} samples...")

        patterns = ["grid", "circles", "stripes", "checkerboard", "random_shapes"]

        for idx in range(self.samples_per_category):
            pattern_type = random.choice(patterns)
            img = Image.new('RGB', self.image_size, 'white')
            draw = ImageDraw.Draw(img)

            if pattern_type == "grid":
                self._draw_grid(draw, spacing=random.randint(20, 80))
            elif pattern_type == "circles":
                self._draw_circles(draw, count=random.randint(5, 30))
            elif pattern_type == "stripes":
                self._draw_stripes(draw, width=random.randint(10, 50))
            elif pattern_type == "checkerboard":
                self._draw_checkerboard(draw, size=random.randint(20, 80))
            else:  # random_shapes
                self._draw_random_shapes(draw, count=random.randint(10, 50))

            self._save_image(img, category, idx, pattern_type=pattern_type)

    def _generate_mixed_samples(self):
        """Generate mixed text and graphics (document-like)."""
        category = "mixed"
        print(f"\nGenerating {category} samples...")

        for idx in range(self.samples_per_category):
            img = Image.new('RGB', self.image_size, 'white')
            draw = ImageDraw.Draw(img)

            # Add some shapes
            for _ in range(random.randint(2, 8)):
                x, y = random.randint(0, 540), random.randint(0, 540)
                w, h = random.randint(50, 150), random.randint(50, 150)
                color = tuple(random.randint(100, 255) for _ in range(3))

                shape = random.choice(['rectangle', 'ellipse'])
                if shape == 'rectangle':
                    draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                else:
                    draw.ellipse([x, y, x+w, y+h], outline=color, width=2)

            # Add text overlay
            text = self._generate_lorem_ipsum(random.randint(30, 100))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            except:
                font = ImageFont.load_default()

            # Draw text in chunks
            y_offset = 20
            words = text.split()
            current_line = []
            for word in words:
                current_line.append(word)
                line_text = ' '.join(current_line)
                bbox = draw.textbbox((0, 0), line_text, font=font)
                if bbox[2] > 600:
                    if len(current_line) > 1:
                        current_line.pop()
                        draw.text((20, y_offset), ' '.join(current_line), fill='black', font=font)
                        current_line = [word]
                        y_offset += 20
                    if y_offset > 600:
                        break

            if current_line and y_offset < 600:
                draw.text((20, y_offset), ' '.join(current_line), fill='black', font=font)

            self._save_image(img, category, idx, text_words=len(words), shape_count=random.randint(2, 8))

    def _generate_gradient_samples(self):
        """Generate images with color gradients."""
        category = "gradient"
        print(f"\nGenerating {category} samples...")

        for idx in range(self.samples_per_category):
            img = Image.new('RGB', self.image_size, 'white')
            pixels = img.load()

            # Random gradient direction
            direction = random.choice(['horizontal', 'vertical', 'diagonal', 'radial'])
            color1 = tuple(random.randint(0, 255) for _ in range(3))
            color2 = tuple(random.randint(0, 255) for _ in range(3))

            for i in range(self.image_size[0]):
                for j in range(self.image_size[1]):
                    if direction == 'horizontal':
                        t = i / self.image_size[0]
                    elif direction == 'vertical':
                        t = j / self.image_size[1]
                    elif direction == 'diagonal':
                        t = (i + j) / (self.image_size[0] + self.image_size[1])
                    else:  # radial
                        cx, cy = self.image_size[0] // 2, self.image_size[1] // 2
                        dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                        t = min(dist / (self.image_size[0] // 2), 1.0)

                    color = tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))
                    pixels[i, j] = color

            self._save_image(img, category, idx, direction=direction)

    def _generate_multilingual_samples(self):
        """Generate text in different languages/scripts."""
        category = "multilingual"
        print(f"\nGenerating {category} samples...")

        # Sample texts in different scripts
        texts = [
            ("english", "The quick brown fox jumps over the lazy dog. " * 30),
            ("numbers", "1234567890 " * 50 + "Price: $99.99 Date: 2024-01-01 " * 10),
            ("mixed_case", "MiXeD CaSe TeXt WiTh VaRiOuS StYlEs " * 20),
            ("punctuation", "Hello, World! How are you? Fine, thanks. #hashtag @mention " * 15),
            ("symbols", "→ ← ↑ ↓ ★ ☆ ♠ ♣ ♥ ♦ © ® ™ € $ ¥ " * 25),
        ]

        for idx in range(self.samples_per_category):
            script, text = random.choice(texts)

            if HAS_VELLO:
                img_array = self.renderer.render(text, width=640, height=640)
                img = Image.fromarray(img_array)
            else:
                img = self._render_text_pil(text, font_size=16, line_spacing=1.2)

            self._save_image(img, category, idx, script=script)

    # Helper methods
    def _generate_lorem_ipsum(self, word_count: int) -> str:
        """Generate random lorem ipsum text."""
        words = [
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
            "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore", "et", "dolore",
            "magna", "aliqua", "enim", "ad", "minim", "veniam", "quis", "nostrud",
            "exercitation", "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo",
            "consequat", "duis", "aute", "irure", "in", "reprehenderit", "voluptate",
            "velit", "esse", "cillum", "fugiat", "nulla", "pariatur", "excepteur", "sint",
            "occaecat", "cupidatat", "non", "proident", "sunt", "culpa", "qui", "officia",
            "deserunt", "mollit", "anim", "id", "est", "laborum"
        ]
        return ' '.join(random.choice(words) for _ in range(word_count))

    def _render_text_pil(self, text: str, font_size: int = 16, line_spacing: float = 1.2) -> Image.Image:
        """Render text using PIL."""
        img = Image.new('RGB', self.image_size, 'white')
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

        # Simple word wrapping
        words = text.split()
        lines = []
        current_line = []

        for word in words:
            current_line.append(word)
            line_text = ' '.join(current_line)
            bbox = draw.textbbox((0, 0), line_text, font=font)
            if bbox[2] > self.image_size[0] - 40:
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(line_text)
                    current_line = []

        if current_line:
            lines.append(' '.join(current_line))

        # Draw lines
        y_offset = 20
        line_height = int(font_size * line_spacing)
        for line in lines:
            if y_offset > self.image_size[1] - 40:
                break
            draw.text((20, y_offset), line, fill='black', font=font)
            y_offset += line_height

        return img

    def _draw_grid(self, draw, spacing: int):
        """Draw a grid pattern."""
        for i in range(0, self.image_size[0], spacing):
            draw.line([(i, 0), (i, self.image_size[1])], fill='black', width=1)
        for j in range(0, self.image_size[1], spacing):
            draw.line([(0, j), (self.image_size[0], j)], fill='black', width=1)

    def _draw_circles(self, draw, count: int):
        """Draw random circles."""
        for _ in range(count):
            x, y = random.randint(0, 640), random.randint(0, 640)
            r = random.randint(10, 100)
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw.ellipse([x-r, y-r, x+r, y+r], outline=color, width=2)

    def _draw_stripes(self, draw, width: int):
        """Draw stripes pattern."""
        x = 0
        toggle = True
        while x < self.image_size[0]:
            if toggle:
                draw.rectangle([x, 0, x+width, self.image_size[1]], fill='black')
            x += width
            toggle = not toggle

    def _draw_checkerboard(self, draw, size: int):
        """Draw checkerboard pattern."""
        for i in range(0, self.image_size[0], size):
            for j in range(0, self.image_size[1], size):
                if (i // size + j // size) % 2 == 0:
                    draw.rectangle([i, j, i+size, j+size], fill='black')

    def _draw_random_shapes(self, draw, count: int):
        """Draw random shapes."""
        for _ in range(count):
            x, y = random.randint(0, 540), random.randint(0, 540)
            w, h = random.randint(20, 100), random.randint(20, 100)
            color = tuple(random.randint(0, 255) for _ in range(3))

            shape_type = random.choice(['rectangle', 'ellipse', 'line'])
            if shape_type == 'rectangle':
                draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
            elif shape_type == 'ellipse':
                draw.ellipse([x, y, x+w, y+h], outline=color, width=2)
            else:  # line
                x2, y2 = random.randint(0, 640), random.randint(0, 640)
                draw.line([(x, y), (x2, y2)], fill=color, width=2)


if __name__ == "__main__":
    # Generate dataset
    generator = VisualizationDataGenerator(samples_per_category=50)
    metadata = generator.generate_all()

    # Print summary
    print("\n" + "="*60)
    print("Dataset Summary:")
    print("="*60)
    categories = {}
    for item in metadata:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items()):
        print(f"  {cat:15s}: {count:3d} images")
    print(f"  {'TOTAL':15s}: {len(metadata):3d} images")
    print("="*60)
