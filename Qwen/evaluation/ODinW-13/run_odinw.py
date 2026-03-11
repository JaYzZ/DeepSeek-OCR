import os
import sys
import json
import argparse
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Any
from collections import defaultdict, OrderedDict
import torch
import requests
import concurrent.futures
import random

# Add parent directory to path to import shared config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import resolve_path, get_data_path, get_results_path, QWEN3_VL_2B_THINKING

# vLLM imports
from vllm import LLM, SamplingParams

# Import LoRARequest for vLLM LoRA support
try:
    from vllm.v1.engine import LoRARequest
    HAS_LORA_REQUEST = True
except ImportError:
    HAS_LORA_REQUEST = False
    LoRARequest = None
# Note: Image preprocessing now handled by vLLM internally
from transformers import AutoProcessor

# pycocotools imports
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Local imports from refactored files
try:
    from .dataset_utils import load_odinw_config, generate_odinw_jobs
    from .eval_utils import compute_metrics
except ImportError:
    from dataset_utils import load_odinw_config, generate_odinw_jobs
    from eval_utils import compute_metrics

# Set vLLM multiprocessing method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare inputs for vLLM - let vLLM handle everything (official approach).
    """

    # Apply chat template to get prompt with proper placeholders
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text = text + "<|im_start|>assistant\n"

    # Extract raw images from messages - let vLLM handle preprocessing
    raw_images = []
    raw_videos = []

    for item in messages[0].get('content', []):
        if isinstance(item, dict):
            if item.get('type') == 'image':
                raw_images.append(item['image'])  # Can be path, PIL image, or base64
            elif item.get('type') == 'video':
                raw_videos.append(item['video'])

    # Get min/max pixels from processor
    min_pixels = getattr(processor.image_processor, 'min_pixels', 28 * 28 * 256)
    max_pixels = getattr(processor.image_processor, 'max_pixels', 28 * 28 * 2048)

    mm_data = {}
    if raw_images:
        mm_data['image'] = raw_images
    if raw_videos:
        mm_data['video'] = raw_videos

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': {
            'min_pixels': min_pixels,
            'max_pixels': max_pixels,
        }
    }


def run_inference(args):
    """Run inference on the ODinW dataset using vLLM or external API server."""
    # Check for server inference mode
    api_url = getattr(args, 'api_url', None) or os.environ.get('LOCAL_API_URL')

    print("\n" + "="*80)
    if api_url:
        print("🚀 ODinW Inference with External API Server")
        print(f"   API URL: {api_url}")
    else:
        print("🚀 ODinW Inference with vLLM (High-Speed Mode)")
    print("="*80 + "\n")

    # Generate task list (sampling is now handled inside generate_odinw_jobs at image level)
    question_list, datasets = generate_odinw_jobs(args.data_dir, args)
    print(f"✓ Generated {len(question_list)} inference jobs\n")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Set up generation parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        stop_token_ids=[],
    )
    
    print(f"\n⚙️  Generation parameters (vLLM SamplingParams):")
    print(f"   max_tokens={sampling_params.max_tokens}")
    print(f"   temperature={sampling_params.temperature}, top_p={sampling_params.top_p}, top_k={sampling_params.top_k}")
    print(f"   repetition_penalty={sampling_params.repetition_penalty}")
    print(f"   presence_penalty={sampling_params.presence_penalty}")
    print()
    
    results = []
    start_time = time.time()

    if api_url:
        # External server mode: do NOT initialize vLLM locally (GPUs are already occupied by the server).
        api_concurrency = max(1, int(getattr(args, "api_concurrency", 16) or 16))
        print(f"Using external API server (concurrency={api_concurrency})")

        def normalize_messages(messages):
            # Strip file:// from local paths for better compatibility with server preprocessing.
            norm = []
            for msg in messages:
                content = msg.get("content", [])
                if isinstance(content, list):
                    new_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") in ("image", "video"):
                            key = "image" if item.get("type") == "image" else "video"
                            val = item.get(key)
                            if isinstance(val, str) and val.startswith("file://"):
                                item = dict(item)
                                item[key] = val[len("file://"):]
                        new_content.append(item)
                    norm.append({"role": msg.get("role", "user"), "content": new_content})
                else:
                    norm.append({"role": msg.get("role", "user"), "content": content})
            return norm

        def call_api(item):
            payload = {
                "messages": normalize_messages(item["messages"]),
                "max_tokens": int(args.max_new_tokens),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "presence_penalty": float(args.presence_penalty),
                "repetition_penalty": float(args.repetition_penalty),
            }
            last_err = None
            for attempt in range(3):
                try:
                    resp = requests.post(api_url, json=payload, timeout=300)
                    if resp.status_code != 200:
                        last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                        raise RuntimeError(last_err)
                    data = resp.json()
                    text = data["choices"][0]["message"]["content"]
                    return item["question_id"], text, None
                except Exception as e:
                    last_err = str(e)
                    time.sleep((2 ** attempt) + random.random())
            return item["question_id"], None, last_err

        results_by_qid = {}
        if api_concurrency == 1:
            for item in tqdm(question_list, desc="ODinW infer (api)"):
                qid, text, err = call_api(item)
                if err is not None:
                    print(f"Error for question_id {qid}: {err}")
                    continue
                results_by_qid[qid] = text
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=api_concurrency) as ex:
                futs = [ex.submit(call_api, item) for item in question_list]
                for fut in tqdm(concurrent.futures.as_completed(futs), total=len(futs), desc="ODinW infer (api)"):
                    qid, text, err = fut.result()
                    if err is not None:
                        print(f"Error for question_id {qid}: {err}")
                        continue
                    results_by_qid[qid] = text

        for item in question_list:
            qid = item["question_id"]
            if qid not in results_by_qid:
                continue
            response = results_by_qid[qid]
            response_final = str(response).split("</think>")[-1].strip()
            results.append({
                "question_id": qid,
                "annotation": item["annotation"],
                "extra_info": item["extra_info"],
                "result": {"gen": response_final, "gen_raw": response},
                "messages": item["messages"],
            })
    else:
        # Local vLLM mode.
        print(f"Loading processor from {args.model_path}")
        processor = AutoProcessor.from_pretrained(args.model_path)
        print("✓ Processor loaded\n")

        print(f"Initializing vLLM with model: {args.model_path}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   Tensor parallel size: {args.tensor_parallel_size}")

        llm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": args.tensor_parallel_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "trust_remote_code": True,
            "max_model_len": args.max_model_len,
            "limit_mm_per_prompt": {"image": args.max_images_per_prompt},
            "seed": 42,
        }

        if hasattr(args, 'enable_lora') and args.enable_lora and args.lora_path:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = args.max_lora_rank
            llm_kwargs["max_loras"] = 1
            print(f"   LoRA enabled: {args.lora_name} from {args.lora_path}")

        llm = LLM(**llm_kwargs)
        print("✓ vLLM initialized successfully\n")

        print("Preparing inputs for vLLM...")
        all_inputs = []
        for item in tqdm(question_list, desc="Building prompts"):
            vllm_input = prepare_inputs_for_vllm(item['messages'], processor)
            all_inputs.append(vllm_input)
        print(f"✓ Prepared {len(all_inputs)} inputs\n")

        print("="*80)
        print("🚀 Running vLLM batch inference")
        print("="*80)

        lora_request = None
        if HAS_LORA_REQUEST and hasattr(args, 'enable_lora') and args.enable_lora and args.lora_path:
            lora_request = LoRARequest(
                lora_name=args.lora_name,
                lora_int_id=1,
                lora_local_path=args.lora_path,
            )
            print(f"   Using LoRA: {args.lora_name}")

        outputs = llm.generate(all_inputs, sampling_params=sampling_params, lora_request=lora_request)

        for item, output in zip(question_list, outputs):
            response = output.outputs[0].text
            response_final = str(response).split("</think>")[-1].strip()
            results.append({
                "question_id": item["question_id"],
                "annotation": item["annotation"],
                "extra_info": item["extra_info"],
                "result": {"gen": response_final, "gen_raw": response},
                "messages": item["messages"],
            })

    end_time = time.time()
    total_time = end_time - start_time
    if results:
        print(f"\n✓ Inference completed in {total_time:.2f} seconds")
        print(f"  Average: {total_time/len(results):.2f} seconds/sample")
        print(f"  Throughput: {len(results)/total_time:.2f} samples/second\n")
    
    # Save results
    print("Saving results...")
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {len(results)}")
    
    # Save dataset config (for evaluation)
    config_output = args.output_file.replace('.jsonl', '_datasets.json')
    with open(config_output, 'w') as f:
        # Convert config for JSON serialization
        datasets_serializable = {}
        for k, v in datasets.items():
            datasets_serializable[k] = {
                'metainfo': v['metainfo'],
                'data_root': v['data_root'],
                'ann_file': v['ann_file'],
                'data_prefix': v['data_prefix']
            }
        json.dump(datasets_serializable, f, indent=2)
    print(f"✓ Dataset config saved to {config_output}")


def run_evaluation(args):
    """Run evaluation on inference results."""
    print("\n" + "="*80)
    print("🎯 ODinW Evaluation")
    print("="*80 + "\n")

    # Load inference results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            results.append(json.loads(line))

    # Apply deterministic sampling if limit is specified
    if hasattr(args, 'limit') and args.limit is not None and args.limit < len(results):
        import hashlib
        print(f"\n{'='*60}")
        print(f"Applying DETERMINISTIC sampling: {args.limit} samples from {len(results)}")
        print(f"{'='*60}")

        # Add hash for deterministic sorting
        for r in results:
            # Create a hash from dataset_name and img_id for better distribution
            dataset_name = r.get('extra_info', {}).get('dataset_name', '')
            img_id = r.get('extra_info', {}).get('img_id', '')
            hash_input = f"{dataset_name}_{img_id}"
            r['_hash'] = hashlib.md5(hash_input.encode()).hexdigest()

        # Sort by hash and take first N
        results = sorted(results, key=lambda x: x['_hash'])
        results = results[:args.limit]

        # Remove the hash field
        for r in results:
            del r['_hash']

        print(f"✓ Sampled {args.limit} examples deterministically")
        print(f"{'='*60}\n")

    print(f"✓ Loaded {len(results)} inference results\n")
    
    # Load dataset config
    config_path = os.path.join(args.data_dir, "odinw13_config.py")
    datasets = load_odinw_config(config_path)
    
    # Group by dataset
    all_outputs = defaultdict(list)
    for job in results:
        all_outputs[job["extra_info"]["dataset_name"]].append(job)
    
    all_results = {}
    
    # Evaluate each dataset
    for dataset_name, sub_jobs in all_outputs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"{'='*60}")
        print(f"Total samples in results: {len(sub_jobs)}")

        anno_path = resolve_path(sub_jobs[0]["extra_info"]["anno_path"])
        coco_api = COCO(anno_path)

        # Load classes from COCO API instead of config
        cat_ids = coco_api.getCatIds()
        cats = coco_api.loadCats(cat_ids)
        classes = [cat['name'] for cat in cats]
        print(f"Classes: {classes[:5]}... (total {len(classes)})")

        # Get all image IDs in the annotation file
        all_img_ids = coco_api.getImgIds()
        print(f"Total images in annotations: {len(all_img_ids)}")
        print(f"Total inference jobs: {len(sub_jobs)}")

        pred_bboxes_per_img = defaultdict(list)
        parse_failures = 0
        empty_predictions = 0

        for job in sub_jobs:
            img_id = job["extra_info"]["img_id"]
            resized_h = job["extra_info"]["resized_h"]
            resized_w = job["extra_info"]["resized_w"]
            img_h = job["extra_info"]["img_h"]
            img_w = job["extra_info"]["img_w"]

            answer = job['result']['gen']

            # Parse predictions - improved to handle Thinking model output
            import re
            import ast

            # Check for text-only "no object" responses and convert to empty array
            no_object_patterns = [
                r'there is no object',
                r'no object belonging',
                r'no objects? found',
                r'does not match the description',
                r'no .*? (in the image|in this image)'
            ]

            answer_lower = answer.lower().strip()
            is_no_object_response = any(re.search(pattern, answer_lower) for pattern in no_object_patterns)

            if is_no_object_response and '```' not in answer and '[' not in answer:
                # Model said "no objects" but didn't output JSON - treat as empty array
                json_data = []
            else:
                # Try to extract JSON from thinking output
                # Pattern 1: Find ```json ... ``` blocks
                json_pattern = r'```json\s*(.*?)\s*```'
                json_matches = re.findall(json_pattern, answer, re.DOTALL)

                # Pattern 2: Find ``` ... ``` blocks (without json label)
                if not json_matches:
                    json_pattern = r'```\s*(.*?)\s*```'
                    json_matches = re.findall(json_pattern, answer, re.DOTALL)

                # Pattern 3: Look for array patterns [...]
                if not json_matches:
                    array_pattern = r'\[\s*\{[^\]]*\}\s*\]'
                    json_matches = re.findall(array_pattern, answer, re.DOTALL)

                # Try each match until we find valid JSON
                json_data = None
                for match in json_matches:
                    try:
                        json_data = ast.literal_eval(match)
                        # Validate that it has the expected structure
                        if isinstance(json_data, list):
                            break
                    except (SyntaxError, ValueError):
                        continue

                # If no JSON found, skip this prediction
                if json_data is None:
                    parse_failures += 1
                    continue

            pred_bboxes = []
            pred_labels = []
            for data in json_data:
                if len(data.get("bbox_2d", [])) != 4:
                    continue
                pred_bboxes.append(data["bbox_2d"])
                # Some models omit the label field; fall back to the known category for this job.
                pred_labels.append(data.get("label") or job.get("extra_info", {}).get("category_name", ""))

            if len(pred_bboxes) == 0:
                empty_predictions += 1
                continue

            if len(pred_bboxes) == 0:
                empty_predictions += 1
                continue
            
            # Coordinate conversion: Model normalizes to 1000x1000
            # Note: Model outputs coordinates in 1000x1000 space regardless of actual image size
            if os.getenv("is_rel", "0") == "1":
                pred_bboxes = np.array(pred_bboxes).reshape(-1, 4) / 1000 * np.array([img_w, img_h, img_w, img_h])
            else:
                # FIX: Model uses 1000x1000 normalization, not resized image coordinates
                if len(pred_bboxes) > 0:
                    pred_bboxes = np.array(pred_bboxes).reshape(-1, 4) / 1000 * np.array([img_w, img_h, img_w, img_h])
                else:
                    pred_bboxes = np.array(pred_bboxes).reshape(-1, 4)
            
            pred_bboxes = pred_bboxes.tolist()
            
            # Group by category
            pred_objs = defaultdict(list)
            for pred_bbox, pred_label in zip(pred_bboxes, pred_labels):
                pred_objs[pred_label].append(pred_bbox)
            
            for k, v in pred_objs.items():
                pred_bboxes_per_img[img_id].append({
                    'label': k,  # Store label string
                    'bbox': v
                })

        print(f"Parsing results: {len(pred_bboxes_per_img)} images with predictions, {parse_failures} failures, {empty_predictions} empty")

        # Prepare evaluation results in COCO format
        pred_results = []
        for k, v in pred_bboxes_per_img.items():
            bboxes = []
            labels = []
            for tmp in v:
                bboxes.extend(tmp['bbox'])
                labels.extend([tmp['label']] * len(tmp['bbox']))

            pred_results.append({
                'img_id': k,
                'bboxes': np.array(bboxes),
                'scores': np.array([1.0] * len(bboxes)),
                'labels': np.array(labels),
            })

        # Compute metrics using COCO API
        if len(pred_results) == 0:
            print(f"{dataset_name}: OrderedDict()")
            all_results[dataset_name] = OrderedDict()
            continue

        # Convert predictions to COCO format
        import tempfile
        tmp_dir = tempfile.TemporaryDirectory()
        outfile_prefix = os.path.join(tmp_dir.name, 'results')

        # Map labels to category IDs (already loaded above)
        label_to_catid = {cat['name'].lower(): cat['id'] for cat in cats}
        print(f"Category mapping: {dict(list(label_to_catid.items())[:5])}...")

        # Check if fuzzy matching is enabled
        use_fuzzy = getattr(args, 'fuzzy_match', False)

        if use_fuzzy:
            from difflib import get_close_matches
            class_names_lower = [cat['name'].lower() for cat in cats]

            def match_label(pred_label: str) -> tuple:
                """Match predicted label to dataset class using exact or fuzzy matching."""
                pred_label_lower = pred_label.lower().strip()

                # Try exact match first
                if pred_label_lower in label_to_catid:
                    return label_to_catid[pred_label_lower], 'exact'

                # Try fuzzy matching for common variations
                close_matches = get_close_matches(pred_label_lower, class_names_lower, n=1, cutoff=0.6)
                if close_matches:
                    return label_to_catid[close_matches[0]], 'fuzzy'

                # Try substring matching (e.g., "rabbit" → "cottontail-rabbit")
                for class_name in class_names_lower:
                    if pred_label_lower in class_name or class_name in pred_label_lower:
                        return label_to_catid[class_name], 'substring'

                return None, 'nomatch'
        else:
            def match_label(pred_label: str) -> tuple:
                """Match predicted label to dataset class using exact matching only."""
                pred_label_lower = pred_label.lower().strip()

                if pred_label_lower in label_to_catid:
                    return label_to_catid[pred_label_lower], 'exact'

                return None, 'nomatch'

        # Build COCO-format predictions
        coco_preds = []
        skipped_labels = 0
        fuzzy_matches = 0
        substring_matches = 0

        for pred in pred_results:
            img_id = pred['img_id']
            for i, (bbox, label, score) in enumerate(zip(pred['bboxes'], pred['labels'], pred['scores'])):
                cat_id, match_type = match_label(label)

                if cat_id is not None:
                    # Convert bbox from xyxy to xywh format
                    x1, y1, x2, y2 = bbox
                    coco_preds.append({
                        'image_id': img_id,
                        'category_id': cat_id,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'score': float(score)
                    })
                    if match_type == 'fuzzy':
                        fuzzy_matches += 1
                    elif match_type == 'substring':
                        substring_matches += 1
                else:
                    skipped_labels += 1

        if use_fuzzy:
            print(f"Built {len(coco_preds)} COCO predictions (exact match), {fuzzy_matches} fuzzy matches, {substring_matches} substring matches, skipped {skipped_labels} due to no match")
        else:
            print(f"Built {len(coco_preds)} COCO predictions (exact match only), skipped {skipped_labels} due to label mismatch")
        print(f"Sample prediction (first): {coco_preds[0] if coco_preds else 'None'}")

        # Load predictions into COCO
        try:
            coco_dt = coco_api.loadRes(coco_preds)
        except IndexError:
            print(f"{dataset_name}: OrderedDict()")
            all_results[dataset_name] = OrderedDict()
            tmp_dir.cleanup()
            continue

        # Run COCO evaluation
        coco_eval = COCOeval(coco_api, coco_dt, 'bbox')
        coco_eval.params.imgIds = list(pred_bboxes_per_img.keys())
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics
        stats = coco_eval.stats
        eval_results = OrderedDict({
            'mAP': stats[0],
            'mAP_50': stats[1],
            'mAP_75': stats[2],
            'mAP_s': stats[3],
            'mAP_m': stats[4],
            'mAP_l': stats[5],
        })
        print(f"{dataset_name}: {eval_results}")
        all_results[dataset_name] = eval_results

        tmp_dir.cleanup()
    
    # Summarize results
    results_ordered = OrderedDict(sorted(all_results.items(), key=lambda x: x[0]))
    metric_items = ['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l']
    results_display = []
    
    for prefix, result in results_ordered.items():
        results_display.append([prefix] + [result.get(k, 0.0) for k in metric_items])
    
    # Calculate average
    average_scores = []
    for col_idx in range(len(metric_items)):
        average_scores.append(np.mean([line[col_idx + 1] for line in results_display]))
    results_display.append(['Average'] + average_scores)
    
    # Print results table
    try:
        from tabulate import tabulate
        print("\n" + "="*80)
        print(
            tabulate(
                results_display,
                headers=["ODinW13 Dataset"] + metric_items,
                tablefmt="fancy_outline",
                floatfmt=".3f",
            )
        )
        print("="*80 + "\n")
    except ImportError:
        print("\n" + "="*80)
        print("ODinW13 Results:")
        print("="*80)
        for row in results_display:
            print(row)
        print("="*80 + "\n")
    
    # Save results
    all_results.update({"Average": average_scores[0]})
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    
    print(f"✓ Evaluation results saved to {args.output_file}")
    print(f"\n{'='*80}")
    print(f"Final Average mAP: {average_scores[0]:.4f}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="ODinW Evaluation with vLLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--model-path", type=str, default=str(QWEN3_VL_2B_THINKING),
                           help="Path to the model")
    infer_parser.add_argument("--data-dir", type=str, default=get_data_path("ODinW-13/odinw"),
                             help="Path to ODinW data directory (containing odinw13_config.py)")
    infer_parser.add_argument("--output-file", type=str, required=True,
                             help="Output file path (relative to evaluation root or absolute)")
    infer_parser.add_argument("--limit", type=int, default=None,
                             help="Limit inference to N samples (deterministic sampling)")

    # vLLM specific parameters
    # Server inference parameters
    infer_parser.add_argument("--api-url", type=str, default=None,
                            help="Use external API server for inference (e.g., http://localhost:8016/v1/chat/completions)")
    infer_parser.add_argument("--api-concurrency", type=int, default=32,
                            help="Concurrent in-flight requests when using --api-url (default: 32)")

    infer_parser.add_argument("--tensor-parallel-size", type=int, default=None,
                            help="Tensor parallel size (default: number of GPUs)")
    infer_parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                            help="GPU memory utilization (0.0-1.0, default: 0.9)")
    infer_parser.add_argument("--max-model-len", type=int, default=128000,
                            help="Maximum model context length (default: 128000)")
    infer_parser.add_argument("--max-images-per-prompt", type=int, default=10,
                            help="Maximum images per prompt (default: 10)")

    # LoRA parameters
    infer_parser.add_argument("--enable-lora", action="store_true",
                            help="Enable LoRA adapter")
    infer_parser.add_argument("--lora-path", type=str, default=None,
                            help="Path to LoRA adapter")
    infer_parser.add_argument("--lora-name", type=str, default="default",
                            help="LoRA adapter name (default: default)")
    infer_parser.add_argument("--max-lora-rank", type=int, default=64,
                            help="Maximum LoRA rank (default: 64)")

    # Generation parameters
    infer_parser.add_argument("--max-new-tokens", type=int, default=8192,
                            help="Maximum number of tokens to generate (default: 8192)")
    infer_parser.add_argument("--temperature", type=float, default=0.7,
                            help="Temperature for sampling (default: 0.7)")
    infer_parser.add_argument("--top-p", type=float, default=0.8,
                            help="Top-p for sampling (default: 0.8)")
    infer_parser.add_argument("--top-k", type=int, default=20,
                            help="Top-k for sampling (default: 20)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                            help="Repetition penalty (default: 1.0)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                            help="Presence penalty (default: 1.5)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, default=get_data_path("ODinW-13/odinw"),
                           help="Path to ODinW data directory (containing odinw13_config.py)")
    eval_parser.add_argument("--input-file", type=str, required=True,
                           help="Input file with inference results (relative to evaluation root or absolute)")
    eval_parser.add_argument("--output-file", type=str, required=True,
                           help="Output file path")
    eval_parser.add_argument("--limit", type=int, default=None,
                           help="Limit evaluation to N samples (deterministic sampling)")
    eval_parser.add_argument("--fuzzy-match", action="store_true",
                           help="Enable fuzzy label matching (default: exact match only)")
    
    args = parser.parse_args()
    
    # Automatically set tensor_parallel_size
    if args.command == 'infer' and args.tensor_parallel_size is None:
        args.tensor_parallel_size = torch.cuda.device_count()
        print(f"Auto-set tensor_parallel_size to {args.tensor_parallel_size}")
    
    if args.command == 'infer':
        run_inference(args)
    elif args.command == 'eval':
        run_evaluation(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
