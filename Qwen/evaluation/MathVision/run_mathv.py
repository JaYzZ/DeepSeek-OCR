import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import time
import re
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import warnings
import traceback

# Add parent directory to path to import shared config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import resolve_path, get_data_path, get_results_path, QWEN3_VL_2B_THINKING

# vLLM imports
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# Note: Image preprocessing now handled by vLLM internally

# Import LoRARequest for vLLM LoRA support
try:
    from vllm.v1.engine import LoRARequest
    HAS_LORA_REQUEST = True
except ImportError:
    HAS_LORA_REQUEST = False
    LoRARequest = None

# Local imports from refactored files.
# Support both:
# - Running as a script from this folder (expects local imports)
# - Importing as a package module (e.g. MathVision.run_mathv)
try:
    from .dataset_utils import load_dataset, dump_image
    from .eval_utils import build_judge, eval_single_sample, MATH_V_acc
except ImportError:
    from dataset_utils import load_dataset, dump_image
    from eval_utils import build_judge, eval_single_sample, MATH_V_acc

# Set vLLM multiprocessing method
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

def clean_for_excel(val):
    """
    Remove characters that are illegal in Excel cells.
    Excel doesn't support control characters (0x00-0x1F) except tab, newline, carriage return.
    """
    if isinstance(val, str):
        # Remove control characters (0x00-0x1F) except tab(0x09), newline(0x0A), carriage return(0x0D)
        return re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', '', val)
    return val

def clean_dataframe_for_excel(df):
    """Clean all string columns in a DataFrame for Excel compatibility."""
    return df.applymap(clean_for_excel) if hasattr(df, 'applymap') else df.map(clean_for_excel)

def build_mathv_prompt(line, dump_image_func, dataset):
    """
    Build MathVision dataset prompt.
    """
    # Standard resolution (MathVision uses smaller min_pixels)
    MIN_PIXELS = 768*28*28  # ~0.6M pixels
    MAX_PIXELS = 5120*28*28  # ~4M pixels
    
    tgt_path = dump_image_func(line)
    question = line['question']
    
    # Build messages in standard conversation format
    content = []
    
    # Add all images first
    if isinstance(tgt_path, list):
        for p in tgt_path:
            content.append({
                "type": "image",
                "image": p,
                "min_pixels": MIN_PIXELS,
                "max_pixels": MAX_PIXELS
            })
    else:
        content.append({
            "type": "image", 
            "image": tgt_path,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS
        })
    
    # Add question text last
    content.append({"type": "text", "text": question})
    
    # Return messages in standard conversation format
    messages = [{
        "role": "user",
        "content": content
    }]
    
    return messages

def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare inputs for vLLM - let vLLM handle everything (official approach).

    Args:
        messages: List of messages in standard conversation format
        processor: AutoProcessor instance

    Returns:
        dict: Input format required by vLLM
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
                raw_images.append(item['image'])  # Can be path or PIL image
            elif item.get('type') == 'video':
                raw_videos.append(item['video'])

    # Get min/max pixels from processor or messages
    min_pixels = None
    max_pixels = None

    # Check if messages have explicit pixel settings
    for item in messages[0].get('content', []):
        if isinstance(item, dict) and item.get('type') == 'image':
            if 'min_pixels' in item:
                min_pixels = item['min_pixels']
            if 'max_pixels' in item:
                max_pixels = item['max_pixels']
            break

    # Fallback to processor defaults
    if min_pixels is None:
        min_pixels = getattr(processor.image_processor, 'min_pixels', 28 * 28 * 256)
    if max_pixels is None:
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
    """Run inference on the MathVision dataset using vLLM."""
    print("\n" + "="*80)
    print("🚀 MathVision Inference with vLLM (High-Speed Mode)")
    print("="*80 + "\n")
    
    # Load dataset
    data = load_dataset(args.dataset)
    
    # Limit number of samples if specified
    if args.num_samples is not None and args.num_samples > 0:
        original_len = len(data)
        data = data.iloc[:args.num_samples]
        print(f"✓ Loaded {len(data)} samples from {args.dataset} (limited from {original_len} samples)")
    else:
        print(f"✓ Loaded {len(data)} samples from {args.dataset}")
    
    # Set up image root directory
    img_root = os.path.join(os.environ['LMUData'], 'images', args.dataset)
    os.makedirs(img_root, exist_ok=True)
    
    # Set up dump_image function
    def dump_image_func(line):
        return dump_image(line, img_root)
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Set up CoT prompt if enabled
    cot_prompt = ""
    if args.use_cot:
        cot_prompt = args.cot_prompt if args.cot_prompt else " Let's think step by step."
        print(f"✓ Using CoT prompt: {cot_prompt[:50]}...")

    # Set up generation parameters (vLLM SamplingParams format)
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
    
    if sampling_params.presence_penalty > 0:
        print(f"   ✅ Anti-repetition enabled (presence_penalty={sampling_params.presence_penalty})")
    
    if sampling_params.temperature <= 0.02 and sampling_params.top_k == 1:
        print(f"   ✅ Using FAST greedy-like decoding")
    else:
        print(f"   ⚠️  Using sampling decoding (slower but more diverse)")
    print()

    # Load processor for input preparation
    print(f"Loading processor from {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("✓ Processor loaded\n")
    
    # Initialize vLLM
    print(f"Initializing vLLM with model: {args.model_path}")
    print(f"   GPU count: {torch.cuda.device_count()}")
    print(f"   Tensor parallel size: {args.tensor_parallel_size}")

    # Build LLM kwargs
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "limit_mm_per_prompt": {"image": args.max_images_per_prompt},
        "seed": 42,
    }

    # Add LoRA parameters if enabled
    if hasattr(args, 'enable_lora') and args.enable_lora and args.lora_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        llm_kwargs["max_loras"] = 1
        # Note: LoRA modules loaded dynamically via LoRARequest during generate()
        print(f"   LoRA enabled: {args.lora_name} from {args.lora_path}")

    llm = LLM(**llm_kwargs)
    print("✓ vLLM initialized successfully\n")
    
    # Prepare all inputs
    print("Preparing inputs for vLLM...")
    all_inputs = []
    all_line_dicts = []
    all_messages = []
    
    for idx, (_, line) in enumerate(tqdm(data.iterrows(), total=len(data), desc="Building prompts")):
        # Convert line to dict
        line_dict = line.to_dict()
        for k, v in line_dict.items():
            if isinstance(v, np.integer):
                line_dict[k] = int(v)
            elif isinstance(v, np.floating):
                line_dict[k] = float(v)
        
        # Build prompt
        messages = build_mathv_prompt(line, dump_image_func, args.dataset)
        
        # Add CoT prompt
        if args.use_cot and len(messages) > 0 and len(messages[0]['content']) > 0:
            last_content = messages[0]['content'][-1]
            if last_content['type'] == 'text':
                last_content['text'] += cot_prompt
        
        # Prepare input for vLLM
        vllm_input = prepare_inputs_for_vllm(messages, processor)
        
        all_inputs.append(vllm_input)
        all_line_dicts.append(line_dict)
        all_messages.append(messages)
    
    print(f"✓ Prepared {len(all_inputs)} inputs\n")

    # Batch inference (vLLM automatic optimization)
    print("="*80)
    print("🚀 Running vLLM batch inference (automatic optimization)")
    print("="*80)
    start_time = time.time()

    # Prepare LoRA request if LoRA is enabled
    lora_request = None
    if HAS_LORA_REQUEST and hasattr(args, 'enable_lora') and args.enable_lora and args.lora_path:
        lora_request = LoRARequest(
            lora_name=args.lora_name,
            lora_int_id=1,
            lora_path=args.lora_path,
        )
        print(f"   Using LoRA: {args.lora_name}")

    outputs = llm.generate(all_inputs, sampling_params=sampling_params, lora_request=lora_request)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n✓ Inference completed in {total_time:.2f} seconds")
    print(f"  Average: {total_time/len(data):.2f} seconds/sample")
    print(f"  Throughput: {len(data)/total_time:.2f} samples/second\n")
    
    # Save results
    print("Saving results...")
    results = []
    
    for idx, (line_dict, messages, output) in enumerate(zip(all_line_dicts, all_messages, outputs)):
        response = output.outputs[0].text
        index = line_dict['index']

        response_final = str(response).split("</think>")[-1].strip()
        
        result = {
            "question_id": int(index) if isinstance(index, np.integer) else index,
            "annotation": line_dict,
            "task": args.dataset,
            "result": {"gen": response_final, "gen_raw": response},
            "messages": messages
        }
        results.append(result)
    
    # Write final results
    with open(args.output_file, 'w') as f:
        for res in results:
            f.write(json.dumps(res) + '\n')
    
    print(f"\n✓ Results saved to {args.output_file}")
    print(f"✓ Total samples processed: {len(results)}")

def run_evaluation(args):
    """Run evaluation on inference results."""
    # Load results
    results = []
    with open(args.input_file, 'r') as f:
        for line in f:
            job = json.loads(line)
            annotation = job["annotation"]
            annotation["prediction"] = job["result"]["gen"]
            results.append(annotation)

    # Apply deterministic sampling if limit is specified
    if hasattr(args, 'limit') and args.limit is not None and args.limit < len(results):
        import hashlib
        print(f"\n{'='*60}")
        print(f"Applying DETERMINISTIC sampling: {args.limit} samples from {len(results)}")
        print(f"{'='*60}")

        # Create hash for deterministic sorting
        for r in results:
            hash_input = f"{r.get('index', '')}{r.get('question', '')}"
            r['_hash'] = hashlib.md5(hash_input.encode()).hexdigest()

        # Sort by hash and take first N
        results = sorted(results, key=lambda x: x['_hash'])
        results = results[:args.limit]

        # Remove hash field
        for r in results:
            del r['_hash']

        print(f"✓ Sampled {args.limit} examples deterministically")
        print(f"{'='*60}\n")

    data = pd.DataFrame.from_records(results)
    data = data.sort_values(by='index')
    data['prediction'] = [str(x) for x in data['prediction']]

    # Load dataset for validation
    meta = load_dataset(args.dataset)

    # Validation
    print(f"len(data): {len(data)}")
    print(f"len(meta): {len(meta)}")
    meta_q_map = {x: y for x, y in zip(meta['index'], meta['question'])}
    data_map = {x: y for x, y in zip(data['index'], data['question'])}
    for k in data_map:
        assert k in meta_q_map, (
            f'eval_file should be the same as or a subset of dataset {args.dataset}'
        )

    # Save intermediate results
    # Must replace .jsonl before .json to avoid .jsonl -> .xlsxl bug
    if args.output_file.endswith('.csv'):
        output_xlsx = args.output_file.replace('.csv', '.xlsx')
    elif args.output_file.endswith('.jsonl'):
        output_xlsx = args.output_file.replace('.jsonl', '.xlsx')
    elif args.output_file.endswith('.json'):
        output_xlsx = args.output_file.replace('.json', '.xlsx')
    else:
        output_xlsx = args.output_file
    clean_dataframe_for_excel(data).to_excel(output_xlsx, index=False)
    print(f"✓ Saved intermediate results to {output_xlsx}")

    # Build judge model
    model = build_judge(
        model=getattr(args, 'eval_model', 'gpt-4o'),
        api_type=getattr(args, 'api_type', 'dash'),
        api_url=getattr(args, 'api_url', None),
        api_key=getattr(args, 'api_key', None)
    )
    
    # Prepare evaluation tasks
    eval_tasks = []
    for i in range(len(data)):
        item = data.iloc[i]
        eval_tasks.append((model, item))
    
    # Run evaluation
    eval_results = []
    
    # Debug mode: process single-threaded with first few samples
    debug = os.environ.get('DEBUG', '').lower() == 'true'
    if debug:
        print("Running in debug mode with first 5 samples...")
        for task in eval_tasks[:5]:
            try:
                result = eval_single_sample(task)
                eval_results.append(result)
            except Exception as e:
                print(f"Error processing task: {e}")
                print(f"Task details: {task}")
                raise
    else:
        # Normal mode: process all samples with threading
        from concurrent.futures import ThreadPoolExecutor
        nproc = getattr(args, 'nproc', 4)
        with ThreadPoolExecutor(max_workers=nproc) as executor:
            for result in tqdm(executor.map(eval_single_sample, eval_tasks), 
                             total=len(eval_tasks), desc="Evaluating"):
                eval_results.append(result)
    
    # Update data with evaluation results
    data['res'] = [r['res'] for r in eval_results]
    data['log'] = [r['log'] for r in eval_results]
    data['extract_model'] = [r['extract_model'] for r in eval_results]
    data['extract_flag'] = [r['extract_flag'] for r in eval_results]
    
    # Save evaluation results
    # Must replace .jsonl before .json to avoid .jsonl -> .xlsxl bug
    if args.output_file.endswith('.csv'):
        storage = args.output_file.replace('.csv', '_eval.xlsx')
    elif args.output_file.endswith('.jsonl'):
        storage = args.output_file.replace('.jsonl', '_eval.xlsx')
    elif args.output_file.endswith('.json'):
        storage = args.output_file.replace('.json', '_eval.xlsx')
    else:
        storage = args.output_file + '_eval.xlsx'
    clean_dataframe_for_excel(data).to_excel(storage, index=False)
    print(f"✓ Saved evaluation results to {storage}")
    
    # Calculate accuracy
    score = MATH_V_acc(storage)
    score_pth = storage.replace('.xlsx', '_score.csv')
    score.to_csv(score_pth, index=False)
    print(f"✓ Saved score to {score_pth}")
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"{'='*50}")
    print(score)
    print(f"{'='*50}\n")
    
    return score

def main():
    parser = argparse.ArgumentParser(description="MathVision Evaluation with vLLM")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Inference parser
    infer_parser = subparsers.add_parser("infer", help="Run inference with vLLM")
    infer_parser.add_argument("--model-path", type=str, default=str(QWEN3_VL_2B_THINKING),
                           help="Path to the model")
    infer_parser.add_argument("--dataset", type=str, default="MathVision", 
                            choices=["MathVision", "MathVision_MINI"],
                            help="Dataset name")
    infer_parser.add_argument("--data-dir", type=str, default=get_data_path("MathVision"),
                           help="Path to MathVision data directory")
    infer_parser.add_argument("--output-file", type=str, required=True,
                           help="Output file path (relative to evaluation root or absolute)")
    infer_parser.add_argument("--num-samples", type=int, default=None, 
                            help="Number of samples to process (default: None, process all samples)")
    infer_parser.add_argument("--use-cot", action="store_true", help="Use Chain-of-Thought prompting")
    infer_parser.add_argument("--cot-prompt", type=str, default="", help="Custom Chain-of-Thought prompt")
    
    # vLLM specific parameters
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
    infer_parser.add_argument("--max-new-tokens", type=int, default=32768, 
                            help="Maximum number of tokens to generate (default: 2048)")
    infer_parser.add_argument("--temperature", type=float, default=0.7, 
                            help="Temperature for sampling (default: 0.7 for greedy-like decoding)")
    infer_parser.add_argument("--top-p", type=float, default=0.8, 
                            help="Top-p for sampling (default: 0.8 for greedy-like decoding)")
    infer_parser.add_argument("--top-k", type=int, default=20, 
                            help="Top-k for sampling (default: 20 for greedy decoding)")
    infer_parser.add_argument("--repetition-penalty", type=float, default=1.0,
                            help="Repetition penalty (default: 1.0, increase to 1.2-1.5 to reduce repetition)")
    infer_parser.add_argument("--presence-penalty", type=float, default=1.5,
                            help="Presence penalty (default: 1.5, range: 0.0-2.0, penalize tokens that have already appeared)")
    
    # Evaluation parser
    eval_parser = subparsers.add_parser("eval", help="Run evaluation")
    eval_parser.add_argument("--data-dir", type=str, default=get_data_path("MathVision"),
                           help="Path to MathVision data directory")
    eval_parser.add_argument("--input-file", type=str, required=True,
                           help="Input file with inference results (relative to evaluation root or absolute)")
    eval_parser.add_argument("--output-file", type=str, required=True,
                           help="Output file path (relative to evaluation root or absolute)")
    eval_parser.add_argument("--limit", type=int, default=None,
                           help="Limit evaluation to N samples (deterministic sampling)")
    eval_parser.add_argument("--dataset", type=str, default="MathVision",
                            choices=["MathVision", "MathVision_MINI"],
                            help="Dataset name")
    eval_parser.add_argument("--eval-model", type=str, default="gpt-4o",
                            help="Model to use for evaluation (default: gpt-4o)")
    eval_parser.add_argument("--api-type", type=str, default="custom", choices=["custom", "local", "dash", "mit"],
                            help="API type for evaluation (default: local)")
    eval_parser.add_argument("--api-url", type=str, default=None,
                            help="API URL for local API (default: http://localhost:8016/v1/chat/completions)")
    eval_parser.add_argument("--api-key", type=str, default=None,
                            help="API key for local API (default: EMPTY)")
    eval_parser.add_argument("--nproc", type=int, default=4, help="Number of processes to use")
    
    args = parser.parse_args()
    
    # Set data directory if provided
    if hasattr(args, 'data_dir') and args.data_dir:
        os.environ['LMUData'] = args.data_dir
    
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
