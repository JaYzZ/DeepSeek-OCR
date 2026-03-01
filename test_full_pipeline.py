#!/usr/bin/env python
"""Test the full training pipeline from raw data to loss calculation.

This tests:
1. Raw data loading from disk
2. Collator processing (full pipeline)
3. Batch structure creation
4. Loss calculation with collator output
"""

import sys
import json
import torch
sys.path.insert(0, '/share/project/xiyan/sources/DeepSeek-OCR')

import os
os.environ['QWEN3VL_LATENT_TOKEN_ID'] = '151669'
os.environ['QWEN3VL_THINKING_START_ID'] = '151667'
os.environ['QWEN3VL_THINKING_END_ID'] = '151668'
os.environ['QWEN3VL_CUTOFF_LEN'] = '8192'

from Qwen.llamafactory import integration as lfi
from Qwen.llamafactory.integration import LatentVAE
from transformers import AutoTokenizer, AutoProcessor

print("=== Full Training Pipeline Test ===")

# Load tokenizer and processor
model_path = '/share/project/xiyan/sources/DeepSeek-OCR/Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking'
print("\n1. Loading tokenizer and processor...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

latent_token_str = "<latent>"
if latent_token_str not in tokenizer.get_vocab():
    tokenizer.add_tokens([latent_token_str])
latent_token_id = tokenizer.convert_tokens_to_ids(latent_token_str)
print(f"   <latent> token ID: {latent_token_id}")

# Load raw samples from disk
print("\n2. Loading raw data from disk...")
raw_samples = []
with open('Qwen/data/r1_onevision_thinking.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 4:  # Load 4 samples for batch testing
            break
        raw_samples.append(json.loads(line))

print(f"   Loaded {len(raw_samples)} raw samples")

# Simulate what the dataset does - tokenize and prepare
print("\n3. Tokenizing samples...")
tokenized_samples = []
for sample in raw_samples:
    # Find assistant content with latent
    assistant_content = ""
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant' and '<latent>' in msg.get('content', ''):
            assistant_content = msg.get('content', '')
            break

    if not assistant_content:
        continue

    # Tokenize
    input_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
    labels = input_ids.copy()

    # Get latent paths from sample
    latent_paths = sample.get('latent_ground_truth', [])
    latent_supervision_paths = sample.get('latent_supervision', [])

    tokenized_samples.append({
        'input_ids': input_ids,
        'labels': labels,
        'latent_ground_truth': latent_paths,
        'latent_supervision': latent_supervision_paths,
    })

print(f"   Tokenized {len(tokenized_samples)} samples")

# Test expansion function (what collator does)
print("\n4. Testing expansion (simulating collator pack)...")
expanded_samples = []
for sample in tokenized_samples:
    input_ids = torch.tensor(sample['input_ids'], dtype=torch.long)
    labels = torch.tensor(sample['labels'], dtype=torch.long)

    # Load latent tensors
    latent_tensors = lfi._load_latent_tensors(sample['latent_ground_truth'])
    latent_sup_tensors = lfi._load_latent_tensors(sample['latent_supervision']) if sample['latent_supervision'] else latent_tensors

    new_input_ids, new_labels = lfi._expand_sample_for_latent_injection(
        input_ids=input_ids,
        labels=labels,
        latent_token_id=latent_token_id,
        thinking_start_id=151667,
        thinking_end_id=151668,
        latent_ground_truth=latent_tensors,
        ignore_index=-100,
        cot_sampled_token_ids=None,
    )

    expanded_samples.append({
        'input_ids': new_input_ids,
        'labels': new_labels,
        'latent_ground_truth': [latent_tensors],  # Wrap in list like dataset does
        'latent_supervision': [latent_sup_tensors],
    })

    print(f"   Sample: {len(sample['input_ids'])} -> {len(new_input_ids)} tokens")

# Simulate collator creating nested structure
print("\n5. Simulating collator batch structure...")

# This is what _add_latent_supervision_to_batch produces - ALREADY FLATTENED
batch_latent_ground_truth = []
batch_latent_supervision = []

for sample in expanded_samples:
    # Flatten ONCE in collator: [[tensor]] -> [tensor]
    flat_gt = lfi._flatten_latent_tensors(sample['latent_ground_truth'])
    flat_sup = lfi._flatten_latent_tensors(sample['latent_supervision'])
    batch_latent_ground_truth.append(flat_gt)
    batch_latent_supervision.append(flat_sup)

print(f"   batch_latent_ground_truth structure: {len(batch_latent_ground_truth)} samples")
print(f"   Sample 0 is flat list: {type(batch_latent_ground_truth[0])}, len={len(batch_latent_ground_truth[0])}")
print(f"   Sample 0[0] type: {type(batch_latent_ground_truth[0][0])}")

# For loss computation, we need to pad all samples to same length
# Find max seq_len
max_seq_len = max(len(s['input_ids']) for s in expanded_samples)
print(f"   Max sequence length: {max_seq_len}")

# Create mock hidden states for loss computation
print("\n6. Testing loss computation with collator structure...")
batch_size = len(expanded_samples)
hidden_dim = 2048

# Create hidden states with max length
mock_hidden = torch.randn(batch_size, max_seq_len, hidden_dim)

# Create latent positions
mock_positions = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
for i, sample in enumerate(expanded_samples):
    # Find latent positions in expanded sequence
    input_ids_list = sample['input_ids']
    for j, tid in enumerate(input_ids_list):
        if tid == latent_token_id:
            mock_positions[i, j] = True

print(f"   Latent positions per sample: {mock_positions.sum(dim=1).tolist()}")

# Test pre_thinking_mse_loss with collator structure
try:
    pre_loss = lfi._compute_pre_thinking_mse_loss(
        hidden_states=mock_hidden,
        latent_ground_truth=batch_latent_ground_truth,
        latent_positions=mock_positions,
    )
    print(f"   ✅ _compute_pre_thinking_mse_loss: {pre_loss.item() if pre_loss else 'None'}")
except Exception as e:
    print(f"   ❌ _compute_pre_thinking_mse_loss failed: {e}")

# Test VAE loss with collator structure
try:
    # Create real LatentVAE
    vae = LatentVAE(hidden_size=hidden_dim, intermediate_size=512, deterministic=False)
    vae_loss, _, _, _, _ = lfi._compute_vae_loss(
        vae=vae,
        hidden_states=mock_hidden,
        latent_supervision=batch_latent_supervision,
        latent_positions=mock_positions,
    )
    print(f"   ✅ _compute_vae_loss: {vae_loss.item() if vae_loss else 'None'}")
except Exception as e:
    print(f"   ❌ _compute_vae_loss failed: {e}")

# Test OT loss with collator structure
try:
    ot_loss, ot_stats = lfi._compute_ot_loss(
        hidden_states=mock_hidden,
        latent_supervision=batch_latent_supervision,
        latent_positions=mock_positions,
    )
    print(f"   ✅ _compute_ot_loss: {ot_loss.item() if ot_loss else 'None'}")
except Exception as e:
    print(f"   ❌ _compute_ot_loss failed: {e}")

# Test REPA loss with collator structure
try:
    repa_loss = lfi._compute_repa_loss(
        hidden_states=mock_hidden,
        latent_supervision=batch_latent_supervision,
        latent_positions=mock_positions,
    )
    print(f"   ✅ _compute_repa_loss: {repa_loss.item() if repa_loss else 'None'}")
except Exception as e:
    print(f"   ❌ _compute_repa_loss failed: {e}")

# Test MSE loss with collator structure
try:
    mse_loss = lfi._compute_mse_loss(
        hidden_states=mock_hidden,
        latent_supervision=batch_latent_supervision,
        latent_positions=mock_positions,
    )
    print(f"   ✅ _compute_mse_loss: {mse_loss.item() if mse_loss else 'None'}")
except Exception as e:
    print(f"   ❌ _compute_mse_loss failed: {e}")

# Test contrastive loss with collator structure
try:
    contrastive_loss = lfi._compute_contrastive_loss(
        hidden_states=mock_hidden,
        latent_supervision=batch_latent_supervision,
        latent_positions=mock_positions,
    )
    print(f"   ✅ _compute_contrastive_loss: {contrastive_loss.item() if contrastive_loss else 'None'}")
except Exception as e:
    print(f"   ❌ _compute_contrastive_loss failed: {e}")

print("\n=== Test 3: Pack Flow Testing (exact training path) ===")

# Test using EXACT same class and methods as training
class MockCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or 0
        self.label_pad_token_id = -100
        self.block_diag_attn = False

class MockLogger:
    def warning(self, msg):
        print(f"   WARNING: {msg}")
    def info(self, msg):
        print(f"   INFO: {msg}")
    def debug(self, msg):
        pass

print("\n7. Testing pack flow with exact training functions...")

# Load raw samples from disk - exactly like dataset does
raw_samples_for_pack = []
with open('Qwen/data/r1_onevision_thinking.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 50:
            break
        raw_samples_for_pack.append(json.loads(line))

# Prepare batch exactly like dataset does - include ALL samples
batch_for_collator = []
for sample in raw_samples_for_pack:
    # Get assistant content
    assistant_content = ""
    for msg in sample.get('messages', []):
        if msg.get('role') == 'assistant':
            assistant_content = msg.get('content', '')
            break

    if not assistant_content:
        continue

    # Tokenize
    input_ids = tokenizer.encode(assistant_content, add_special_tokens=False)
    labels = input_ids.copy()

    # Get latent ground truth from sample
    item = {
        'input_ids': input_ids,
        'labels': labels,
        'latent_ground_truth': sample.get('latent_ground_truth', []),
    }
    batch_for_collator.append(item)

# Create mock collator and logger
collator = MockCollator(tokenizer)
logger = MockLogger()

# EXACT same call as in wrapped_call: _pack_features_after_injection
latent_fields_list_raw = [{'latent_ground_truth': item.get('latent_ground_truth', [])} for item in batch_for_collator]

packed_features, packed_latent_fields = lfi._pack_features_after_injection(
    batch=batch_for_collator,
    latent_fields_list=latent_fields_list_raw,
    collator=collator,
    logger=logger,
)

print(f"   Packed into {len(packed_features)} sequences")

# Verify each packed sequence - same check as in loss functions
mismatch_count = 0
for i, (feature, latent_field) in enumerate(zip(packed_features, packed_latent_fields)):
    input_ids = feature['input_ids']
    latent_count = input_ids.count(latent_token_id)

    # Get gt and calculate total sequence length - exact same as loss functions
    gt = latent_field.get('latent_ground_truth', [])
    gt_flat = lfi._flatten_latent_tensors(gt)
    gt_seq_len = sum(t.shape[0] for t in gt_flat) if gt_flat else 0

    if latent_count != gt_seq_len:
        mismatch_count += 1
        print(f"   ❌ Pack seq {i}: {latent_count} latent vs {gt_seq_len} gt length")
    else:
        print(f"   ✅ Pack seq {i}: {latent_count} latent = {gt_seq_len} gt length")

if mismatch_count == 0:
    print("   ✅ All pack sequences have correct latent/gt matching!")
else:
    print(f"   ❌ {mismatch_count} sequences have mismatch!")

print("\n=== Test 4: Simulate Training Loop with Full Integration ===")

# Test simulating the actual training loop flow
import os
os.environ['QWEN3VL_LOSS_TYPE'] = 'vae+pre_think_mse'  # Match curriculum stage 1

# Verify loss spec
loss_spec = lfi._get_loss_spec()
print(f"   Loss spec: {loss_spec}")
print(f"   'vae' in loss_spec: {'vae' in loss_spec}")

# Test with a simple forward pass simulation
print("\n   Testing loss computation with current loss_spec:")
batch_size = 2
hidden_dim = 2048
seq_len = 500

# Create mock data
mock_hidden = torch.randn(batch_size, seq_len, hidden_dim)
mock_positions = torch.zeros(batch_size, seq_len, dtype=torch.bool)
mock_positions[:, 100:110] = True  # 10 latent positions per sample

# Create latent supervision
mock_latent_sup = [
    [torch.randn(10, 2048) for _ in range(1)]  # 10 latents
    for _ in range(batch_size)
]

# Create VAE
vae = LatentVAE(hidden_size=hidden_dim, intermediate_size=512, deterministic=False)
print(f"   VAE created: {vae}")

# Compute VAE loss
vae_loss, _, _, _, _ = lfi._compute_vae_loss(
    vae=vae,
    hidden_states=mock_hidden,
    latent_supervision=mock_latent_sup,
    latent_positions=mock_positions,
)
print(f"   VAE loss: {vae_loss.item() if vae_loss else 'None'}")

# Compute pre_think_mse loss
mock_latent_gt = [
    [torch.randn(10, 2048) for _ in range(1)]
    for _ in range(batch_size)
]
pre_loss = lfi._compute_pre_thinking_mse_loss(
    hidden_states=mock_hidden,
    latent_ground_truth=mock_latent_gt,
    latent_positions=mock_positions,
)
print(f"   pre_think_mse loss: {pre_loss.item() if pre_loss else 'None'}")

# Test pred_embed_forward_loss (second forward with sampled latents)
print("\n=== Test 5: pred_embed_forward_loss (second forward) ===")

# Create a mock model that mimics PEFT structure with base_model
class MockBaseModel(torch.nn.Module):
    def __init__(self, hidden_dim=2048, vocab_size=151680):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_dim)
        # Mock language model that returns loss
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        # Simple forward: just compute CE loss on labels
        if labels is not None:
            # Create dummy logits
            batch_size, seq_len = labels.shape
            logits = torch.randn(batch_size, seq_len, self.lm_head.out_features, device=labels.device)
            # Mask -100 labels
            labels_for_ce = labels.clone()
            labels_for_ce[labels_for_ce == -100] = 0
            # Simple cross entropy
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels_for_ce.view(-1),
                ignore_index=-100
            )
            # Return mock output with loss
            return type('obj', (object,), {'loss': loss})()
        return type('obj', (object,), {'loss': None})()

class MockPEFTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = type('obj', (object,), {
            'model': MockBaseModel()
        })()

# Create mock model
mock_model = MockPEFTModel()

# Create inputs for pred_embed_forward
mock_inputs_embeds = torch.randn(batch_size, seq_len, hidden_dim)
mock_sampled_latents = torch.randn(20, hidden_dim)  # 10 per sample * 2
mock_batch_indices = torch.tensor([0]*10 + [1]*10)
mock_seq_indices = torch.tensor([99, 100, 101, 102, 103, 104, 105, 106, 107, 108] * 2)
mock_labels = torch.randint(0, 150000, (batch_size, seq_len))
mock_attention_mask = torch.ones(batch_size, seq_len)

# Test pred_embed_forward_loss
try:
    pred_embed_loss = lfi._compute_pred_embed_forward_loss(
        model=mock_model,
        original_inputs_embeds=mock_inputs_embeds,
        sampled_latents=mock_sampled_latents,
        batch_indices=mock_batch_indices,
        seq_indices=mock_seq_indices,
        labels=mock_labels,
        attention_mask=mock_attention_mask,
        pixel_values=None,
        image_grid_thw=None,
    )
    print(f"   ✅ pred_embed_forward_loss: {pred_embed_loss.item() if pred_embed_loss else 'None'}")
except Exception as e:
    print(f"   ❌ pred_embed_forward_loss failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Full Pipeline Test Complete ===")
