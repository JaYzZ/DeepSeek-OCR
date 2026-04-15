"""
ODinW dataset loading and processing utilities.
"""
import os
import math
from typing import Dict, List, Tuple
from pycocotools.coco import COCO


def round_by_factor(number: int, factor: int) -> int:
    """Return the nearest integer divisible by factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the ceiling integer divisible by factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the floor integer divisible by factor."""
    return math.floor(number / factor) * factor


def smart_resize(height: int, width: int, factor: int = 28, 
                 min_pixels: int = 56*56, max_pixels: int = 14*14*4*1280, 
                 max_long_side: int = 8192) -> Tuple[int, int]:
    """Resize image to meet the following conditions:
        1. Both height and width are divisible by factor
        2. Total pixels are within [min_pixels, max_pixels]
        3. Longest side is within max_long_side
        4. Aspect ratio is preserved
    
    Args:
        height: Original image height
        width: Original image width
        factor: Size must be divisible by this factor
        min_pixels: Minimum pixel count
        max_pixels: Maximum pixel count
        max_long_side: Maximum longest side
    
    Returns:
        (resized_height, resized_width): Resized dimensions
    """
    if height < 2 or width < 2:
        raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f'absolute aspect ratio must be smaller than 200, got {height} / {width}')

    if max(height, width) > max_long_side:
        beta = max(height, width) / max_long_side
        height, width = int(height / beta), int(width / beta)

    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    
    return h_bar, w_bar


def load_odinw_config(config_path: str) -> Dict:
    """Load odinw13_config.py configuration file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        datasets: Dictionary mapping dataset names to configurations
    """
    import runpy
    config = runpy.run_path(config_path)
    dataset_configs = config["datasets"]
    dataset_names = config["dataset_prefixes"]
    
    datasets = {}
    for dataset_name, dataset_config in zip(dataset_names, dataset_configs):
        datasets[dataset_name] = dataset_config
    
    return datasets


def generate_odinw_jobs(data_dir: str, args) -> Tuple[List[Dict], Dict]:
    """Generate inference task list for ODinW dataset.

    Args:
        data_dir: Data directory path (containing odinw13_config.py)
        args: Command line arguments (can have 'limit' attribute for sampling)

    Returns:
        (question_list, datasets): Task list and dataset configurations
    """
    # Load config
    config_path = os.path.join(data_dir, "odinw13_config.py")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    datasets = load_odinw_config(config_path)

    # Collect images per dataset
    images_by_dataset = {}  # {dataset_name: [(img_idx, img_meta, dataset, data_config), ...]}
    for data_name, data_config in datasets.items():
        # Build data paths
        idx = data_config["data_root"].find('data/odinw/') + len('data/odinw/')
        sub_root = os.path.join(data_dir, data_config["data_root"][idx:])
        sub_anno = sub_root + data_config["ann_file"]

        # Load COCO format annotations
        dataset = COCO(sub_anno)

        # Collect all images for this dataset
        images_by_dataset[data_name] = []
        for img_idx, img_meta in dataset.imgs.items():
            images_by_dataset[data_name].append((img_idx, img_meta, dataset, data_config))

    # Apply sampling at IMAGE level with per-dataset balancing if limit is specified
    if hasattr(args, 'limit') and args.limit is not None:
        import hashlib
        total_images = sum(len(imgs) for imgs in images_by_dataset.values())
        num_datasets = len(images_by_dataset)

        print(f"\n{'='*60}")
        print(f"Applying DETERMINISTIC sampling: {args.limit} images from {total_images} total")
        print(f"Balancing across {num_datasets} datasets")
        print(f"{'='*60}\n")

        # Calculate images per dataset (proportional allocation)
        images_per_dataset = {}
        remaining = args.limit

        for data_name, images in images_by_dataset.items():
            # Allocate proportionally to dataset size, but ensure at least 1 image per dataset
            if len(images) > 0:
                allocation = max(1, round(args.limit * len(images) / total_images))
                images_per_dataset[data_name] = min(allocation, len(images))

        # If we over-allocated, reduce proportionally from largest datasets
        total_allocated = sum(images_per_dataset.values())
        if total_allocated > args.limit:
            # Reduce from datasets with most images
            while total_allocated > args.limit:
                for data_name in sorted(images_per_dataset.keys(), key=lambda x: -len(images_by_dataset[x])):
                    if images_per_dataset[data_name] > 1 and total_allocated > args.limit:
                        images_per_dataset[data_name] -= 1
                        total_allocated -= 1

        # Sample images from each dataset
        sampled_images = []
        for data_name, num_to_sample in images_per_dataset.items():
            images = images_by_dataset[data_name]

            # Sort images by hash for deterministic sampling
            hashed_images = []
            for img_idx, img_meta, dataset, data_config in images:
                hash_input = f"{data_name}_{img_meta['id']}"
                item_hash = hashlib.md5(hash_input.encode()).hexdigest()
                hashed_images.append((item_hash, img_idx, img_meta, dataset, data_config))

            # Sort and take first N
            hashed_images.sort(key=lambda x: x[0])
            sampled_images.extend([(data_name, img_idx, img_meta, dataset, data_config)
                                  for _, img_idx, img_meta, dataset, data_config in hashed_images[:num_to_sample]])

            print(f"  {data_name}: {num_to_sample} images (from {len(images)} total)")

        print(f"\n✓ Sampled {len(sampled_images)} unique images deterministically (balanced across datasets)\n")

        all_images = sampled_images
    else:
        # No sampling: flatten all images
        all_images = []
        for data_name, images in images_by_dataset.items():
            for img_idx, img_meta, dataset, data_config in images:
                all_images.append((data_name, img_idx, img_meta, dataset, data_config))

    # Second pass: generate jobs from sampled images
    question_list = []
    question_id = 0
    num_questions_per_dataset = {}

    # Calculate image resolution parameters
    patch_size = 16
    merge_base = 2
    pixels_per_token = patch_size * patch_size * merge_base * merge_base
    min_pixels = pixels_per_token * 768
    max_pixels = pixels_per_token * 12800

    # Process sampled images
    for data_name, img_idx, img_meta, dataset, data_config in all_images:
        # Build data paths (need to rebuild for each image since we're iterating)
        idx = data_config["data_root"].find('data/odinw/') + len('data/odinw/')
        sub_root = os.path.join(data_dir, data_config["data_root"][idx:])
        sub_anno = sub_root + data_config["ann_file"]
        sub_img_root = sub_root + data_config["data_prefix"]["img"]

        # Load classes from COCO API
        cat_ids = dataset.getCatIds()
        cats = dataset.loadCats(cat_ids)
        classes = [cat['name'] for cat in cats]

        # Initialize counter for this dataset if needed
        if data_name not in num_questions_per_dataset:
            num_questions_per_dataset[data_name] = 0

        img_name = img_meta["file_name"]
        img_path = sub_img_root + img_name
        img_h = img_meta["height"]
        img_w = img_meta["width"]

        # Calculate resized image dimensions
        resized_h, resized_w = smart_resize(
            img_h, img_w,
            factor=32,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            max_long_side=50000
        )

        # Get annotations
        img_annos = dataset.imgToAnns[img_idx]

        # Create one job per category (category-by-category approach)
        for class_name in classes:
            # Build prompt for single category
            prompt = f"Locate every {class_name} in the image and output the coordinates in JSON format."

            # Build messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            # Use a plain filesystem path for local images (more compatible than file:// URIs).
                            "image": img_path,
                            "min_pixels": min_pixels,
                            "max_pixels": max_pixels
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # Build task item with category field
            item = {
                "question_id": question_id,
                "annotation": img_annos,
                'messages': messages,
                "extra_info": {
                    'dataset_name': data_name,
                    'dataset_config': data_config,
                    'img_id': img_meta["id"],
                    'category_name': class_name,  # NEW: Track which category this job is for
                    'anno_path': sub_anno,
                    'resized_h': resized_h,
                    'resized_w': resized_w,
                    'img_h': img_h,
                    'img_w': img_w,
                    'img_path': img_path
                }
            }
            question_list.append(item)
            question_id += 1
            num_questions_per_dataset[data_name] += 1

    # Print statistics
    for data_name, num_questions in num_questions_per_dataset.items():
        print(f'{data_name}: {num_questions}')
    print(f"Total ODinW questions: {len(question_list)}")
    
    return question_list, datasets
