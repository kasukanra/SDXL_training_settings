#!/usr/bin/env python3
"""
convert_wan21_to_comfy.py

Convert Wan21 full fine-tune models into a single ComfyUI .safetensors checkpoint.
Usage:
    python convert_wan21_to_comfy.py \
        /path/to/transformer_dir \
        /path/to/diffusion_pytorch_model.safetensors.index.json \
        /path/to/output_checkpoint.safetensors
"""
import argparse
import json
import pathlib
import re
import torch
from safetensors.torch import load_file, save_file

#── KEY MAPPINGS ────────────────────────────────────────────────────────────────
KEY_MAPPING = {
    # time & text embeddings
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "condition_embedder.time_proj":             "time_projection.1",

    # head (we will override global scale_shift_table → head.modulation by hand below)
    "scale_shift_table": "modulation",
    "proj_out":           "head.head",

    # feed-forward
    "ffn.net.0.proj": "ffn.0",
    "ffn.net.2":      "ffn.2",

    # self-attn → self_attn
    "attn1.to_q":     "self_attn.q",
    "attn1.to_k":     "self_attn.k",
    "attn1.to_v":     "self_attn.v",
    "attn1.to_out.0": "self_attn.o",
    "attn1.norm_q":   "self_attn.norm_q",
    "attn1.norm_k":   "self_attn.norm_k",

    # cross-attn → cross_attn
    "attn2.to_q":     "cross_attn.q",
    "attn2.to_k":     "cross_attn.k",
    "attn2.to_v":     "cross_attn.v",
    "attn2.to_out.0": "cross_attn.o",
    "attn2.norm_q":   "cross_attn.norm_q",
    "attn2.norm_k":   "cross_attn.norm_k",

    # norm2 → norm3
    "norm2":          "norm3",
}

#── BUILD YOUR REGEX REPLACERS ─────────────────────────────────────────────────
REPLACERS = []
for old, new in KEY_MAPPING.items():
    esc_old = re.escape(old)
    REPLACERS.extend([
        # per-block with .weight/.bias
        (rf"^blocks\.(\d+)\.{esc_old}\.(weight|bias)$", rf"blocks.\1.{new}.\2"),
        # per-block without suffix
        (rf"^blocks\.(\d+)\.{esc_old}$",         rf"blocks.\1.{new}"),
        # global with .weight/.bias
        (rf"^{esc_old}\.(weight|bias)$",           rf"{new}.\1"),
        # global without suffix
        (rf"^{esc_old}$",                           rf"{new}"),
    ])

#── EXTRA KEEPERS ──────────────────────────────────────────────────────────────
EXTRA_KEEP = [
    r"head\.head\.(weight|bias)",
    r"patch_embedding\.(weight|bias)",
    r"text_embedding\.(0|2)\.(weight|bias)",
    r"time_embedding\.(0|2)\.(weight|bias)",
    r"time_projection\.1\.(weight|bias)",
]

#── REMAP A SINGLE KEY ─────────────────────────────────────────────────────────
def remap_key(old_key: str, stats: dict) -> str:
    new_key = old_key
    for pat, rep in REPLACERS:
        new_key, cnt = re.subn(pat, rep, new_key)
        stats['remapped'] += cnt
    # check EXTRA_KEEP
    for ex in EXTRA_KEEP:
        if re.fullmatch(ex, new_key):
            stats['kept'] += 1
            return new_key
    # record unmapped
    if new_key == old_key:
        stats['unmapped'].append(old_key)
    return new_key

#── MAIN CONVERSION ────────────────────────────────────────────────────────────
def convert_diffusers_to_single_safetensors(
    diffusers_dir: str,
    index_json_path: str,
    output_path: str
):
    diffusers_dir = pathlib.Path(diffusers_dir)
    index = json.load(open(index_json_path, 'r'))
    weight_map = index['weight_map']

    # load shards
    shards_cache = {}
    original_tensors = {}
    for orig_key, shard_fn in weight_map.items():
        if shard_fn not in shards_cache:
            shards_cache[shard_fn] = load_file(str(diffusers_dir / shard_fn))
        original_tensors[orig_key] = shards_cache[shard_fn][orig_key]

    # remap keys
    stats = {'remapped': 0, 'kept': 0, 'unmapped': []}
    comfy_state = {}
    for old_key, tensor in original_tensors.items():
        new_key = remap_key(old_key, stats)
        comfy_state[new_key] = tensor

    # fix global modulation
    if 'modulation' in comfy_state:
        comfy_state['head.modulation'] = comfy_state.pop('modulation')

    # dummy conv_in
    sample = next(iter(comfy_state.values()))
    dtype, device = sample.dtype, sample.device
    mc = comfy_state['blocks.0.self_attn.k.weight'].shape[0]
    ic = comfy_state['patch_embedding.weight'].shape[1]
    comfy_state['conv_in.weight'] = torch.zeros(mc, ic, 1, 1, dtype=dtype, device=device)
    comfy_state['conv_in.bias']   = torch.zeros(mc, dtype=dtype, device=device)

    # save
    save_file(comfy_state, output_path)
    print(f"✅ Saved ComfyUI checkpoint to {output_path}")
    print(f"→ {stats['remapped']} replacements, {stats['kept']} keys kept.")
    if stats['unmapped']:
        print(f"⚠ {len(stats['unmapped'])} unmapped keys (up to 20):")
        for k in stats['unmapped'][:20]: print("   ", k)

#── SCRIPT ENTRY ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Wan21 snapshot → ComfyUI .safetensors')
    parser.add_argument(
        'snapshot_dir', help='Directory with diffusion_pytorch_model shards')
    parser.add_argument(
        'index_json', help='Path to diffusion_pytorch_model.safetensors.index.json')
    parser.add_argument(
        'output_path', help='Where to write the .safetensors output')
    args = parser.parse_args()

    convert_diffusers_to_single_safetensors(
        args.snapshot_dir,
        args.index_json,
        args.output_path
    )