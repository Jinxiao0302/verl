# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline W4A8 (MXFP4) weight quantization for Ascend NPU inference with vllm-ascend.

This script reproduces, offline, the exact weight quantization the verl rollout
performs online during a W4A8 refit (``quant_weights`` in
``verl/utils/vllm/vllm_quant_utils.py``): each BF16 weight listed in the
checkpoint's ``quant_model_description.json`` is quantized with
``torch_npu.npu_dynamic_mx_quant`` (pure RTN, no calibration) into packed FP4
(E2M1, two values per uint8 byte) plus an MX E8M0 scale per 32 input channels.
The A8 half of W4A8 is always dynamic at inference time, so nothing activation
related is stored.

The output checkpoint can be served directly (``vllm serve --quantization ascend``
or verl ``actor_rollout_ref.rollout.quantization=ascend`` with
``actor_rollout_ref.rollout.load_format=auto``) -- no refit needed.

Must run on an Ascend NPU machine (torch_npu + CANN).

Usage:
    python scripts/offline_w4a8_quantize.py \
        --input-dir /path/to/bf16_checkpoint \
        --output-dir /path/to/w4a8_checkpoint

    # only list which tensors would be quantized, without running any NPU op:
    python scripts/offline_w4a8_quantize.py --input-dir ... --output-dir ... --dry-run
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

logger = logging.getLogger(__file__)

DESCRIPTION_FILENAME = "quant_model_description.json"
INDEX_FILENAME = "model.safetensors.index.json"

# Metadata keys in the description file that are not per-layer quant types.
DESCRIPTION_METADATA_KEYS = {"group_size", "kv_cache_type", "fa_quant_type", "model_quant_type"}


def load_quantized_weight_names(description_path: Path) -> set[str]:
    """Read quant_model_description.json and return the weight names to quantize.

    Keys are either weight names (``...q_proj.weight``) or module names
    (``...q_proj``); module names are normalized to weight names. Only entries
    whose value marks W4A8 MX quantization are returned.
    """
    with open(description_path, encoding="utf-8") as f:
        description = json.load(f)
    names = set()
    for key, value in description.items():
        if key in DESCRIPTION_METADATA_KEYS or not isinstance(value, str):
            continue
        if "W4A8" not in value.upper():
            continue
        names.add(key if key.endswith(".weight") else key + ".weight")
    return names


def safetensors_supports_e8m0() -> bool:
    """Probe whether this safetensors build can store torch.float8_e8m0fnu.

    The online refit path feeds the E8M0 scale tensor straight into
    ``model.load_weights``, so saving scales in native E8M0 matches it most
    closely. Older safetensors versions lack the dtype; fall back to the raw
    bytes viewed as uint8 (the layout modelslim-produced checkpoints use).
    """
    if not hasattr(torch, "float8_e8m0fnu"):
        return False
    try:
        from safetensors.torch import save

        save({"x": torch.zeros(4, dtype=torch.float8_e8m0fnu)})
        return True
    except Exception:
        return False


def quantize_weight_mxfp4(weight: torch.Tensor, device: str, save_e8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize one BF16 weight to packed FP4 + MX scale, on the NPU.

    Mirrors ``quant_weights`` in verl/utils/vllm/vllm_quant_utils.py exactly:
    same op, same axis, same post-processing (flatten/squeeze of the scale,
    uint8 reinterpretation of the packed FP4 bytes).
    """
    import torch_npu  # noqa: F401 -- importing registers the NPU ops

    w = weight.to(device=device, dtype=torch.bfloat16)
    param_lp, param_scale = torch_npu.npu_dynamic_mx_quant(w, axis=-1, dst_type=torch_npu.float4_e2m1fn_x2)
    param_scale = param_scale.flatten(-2, -1).squeeze(-1)
    param_lp = param_lp.view(torch.uint8)
    param_lp = param_lp.cpu().contiguous()
    param_scale = param_scale.cpu().contiguous()
    if not save_e8m0:
        param_scale = param_scale.view(torch.uint8)
    return param_lp, param_scale


def expand_packed_expert(name: str, tensor: torch.Tensor):
    """Expand a packed 3-D expert tensor into per-expert 2-D (name, tensor) pairs.

    Same math as ``_expand_kimi_packed_w4a8_experts`` in vllm_quant_utils.py:
    ``(E, hidden, 2*inter)`` gate_up_proj -> per-expert w1/w3 ``(inter, hidden)``,
    ``(E, inter, hidden)`` down_proj -> per-expert w2 ``(hidden, inter)``.
    """
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    prefix = base.rsplit(".", 1)[0]  # "...block_sparse_moe.experts"
    for expert_id in range(tensor.shape[0]):
        if base.endswith(".experts.gate_up_proj"):
            intermediate = tensor.shape[-1] // 2
            yield f"{prefix}.{expert_id}.w1.weight", tensor[expert_id][:, :intermediate].transpose(0, 1).contiguous()
            yield f"{prefix}.{expert_id}.w3.weight", tensor[expert_id][:, intermediate:].transpose(0, 1).contiguous()
        else:
            yield f"{prefix}.{expert_id}.w2.weight", tensor[expert_id].transpose(0, 1).contiguous()


def is_packed_expert_tensor(name: str, tensor: torch.Tensor) -> bool:
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    return tensor.dim() == 3 and base.endswith((".experts.gate_up_proj", ".experts.down_proj"))


def quantize_shard(
    tensors: dict[str, torch.Tensor],
    quant_names: set[str],
    device: str,
    save_e8m0: bool,
    dry_run: bool,
) -> tuple[dict[str, torch.Tensor], set[str]]:
    """Quantize one shard's tensors. Returns output tensors and the quantized names."""
    out = {}
    quantized = set()
    for name, tensor in tensors.items():
        if is_packed_expert_tensor(name, tensor):
            expanded = list(expand_packed_expert(name, tensor))
            if any(expanded_name in quant_names for expanded_name, _ in expanded):
                for expanded_name, expanded_tensor in expanded:
                    if expanded_name in quant_names:
                        if dry_run:
                            logger.info(f"[dry-run] quantize {expanded_name} {tuple(expanded_tensor.shape)}")
                            continue
                        param_lp, param_scale = quantize_weight_mxfp4(expanded_tensor, device, save_e8m0)
                        out[expanded_name] = param_lp
                        out[expanded_name + "_scale"] = param_scale
                        quantized.add(expanded_name)
                    else:
                        out[expanded_name] = expanded_tensor
                continue
            logger.warning(f"packed expert tensor {name} has no quantized entries in the description; copied as-is")
            out[name] = tensor
            continue

        if name in quant_names:
            if dry_run:
                logger.info(f"[dry-run] quantize {name} {tuple(tensor.shape)}")
                continue
            param_lp, param_scale = quantize_weight_mxfp4(tensor, device, save_e8m0)
            out[name] = param_lp
            out[name + "_scale"] = param_scale
            quantized.add(name)
        else:
            out[name] = tensor
    return out, quantized


def save_shard(tensors: dict[str, torch.Tensor], path: Path):
    try:
        save_file(tensors, str(path), metadata={"format": "pt"})
    except RuntimeError:
        # safetensors refuses tensors sharing storage (e.g. views of one buffer)
        save_file({k: v.clone() for k, v in tensors.items()}, str(path), metadata={"format": "pt"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=str, required=True, help="BF16 checkpoint directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for the W4A8 checkpoint")
    parser.add_argument(
        "--quant-description",
        type=str,
        default=None,
        help=f"Path to {DESCRIPTION_FILENAME} (default: <input-dir>/{DESCRIPTION_FILENAME})",
    )
    parser.add_argument("--device", type=str, default="npu:0", help="NPU device used for the quant op")
    parser.add_argument("--dry-run", action="store_true", help="Only list tensors that would be quantized")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    description_path = Path(args.quant_description) if args.quant_description else input_dir / DESCRIPTION_FILENAME
    quant_names = load_quantized_weight_names(description_path)
    logger.info(f"{len(quant_names)} weights marked for quantization in {description_path}")
    if not quant_names:
        raise ValueError(f"no W4A8 entries found in {description_path}")

    index_path = input_dir / INDEX_FILENAME
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else None
    if index is not None:
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(p.name for p in input_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no safetensors files found in {input_dir}")
    logger.info(f"{len(shard_files)} safetensors shard(s) to process")

    save_e8m0 = safetensors_supports_e8m0()
    logger.info(f"weight_scale dtype on disk: {'float8_e8m0fnu' if save_e8m0 else 'uint8 (raw E8M0 bytes)'}")

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_map = {}
    total_size = 0
    all_quantized: set[str] = set()
    for shard_name in shard_files:
        tensors = load_file(str(input_dir / shard_name))
        out, quantized = quantize_shard(tensors, quant_names, args.device, save_e8m0, args.dry_run)
        all_quantized |= quantized
        if args.dry_run:
            continue
        for name, tensor in out.items():
            weight_map[name] = shard_name
            total_size += tensor.nbytes
        save_shard(out, output_dir / shard_name)
        logger.info(f"{shard_name}: {len(tensors)} tensors in, {len(out)} out, {len(quantized)} quantized")

    missing = quant_names - all_quantized
    if missing:
        logger.warning(f"{len(missing)} description entries matched no checkpoint tensor, e.g. {sorted(missing)[:5]}")
    logger.info(f"quantized {len(all_quantized)} / {len(quant_names)} weights")

    if args.dry_run:
        return

    if index is not None:
        (output_dir / INDEX_FILENAME).write_text(
            json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )

    for item in input_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors" and item.name != INDEX_FILENAME:
            shutil.copy2(item, output_dir / item.name)

    logger.info(f"W4A8 checkpoint written to {output_dir}")
    logger.info(
        "serve it with `vllm serve --quantization ascend` (or verl "
        "actor_rollout_ref.rollout.quantization=ascend) and a real load_format=auto; no refit needed"
    )


if __name__ == "__main__":
    main()
