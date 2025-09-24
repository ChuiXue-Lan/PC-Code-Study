import argparse
import os
import re
from typing import Dict, Iterable, Tuple

import torch


def load_state_dict(checkpoint_path: str) -> Tuple[dict, Dict[str, torch.Tensor]]:
    """Load a checkpoint and return (raw_checkpoint, state_dict_like).

    This function tries common keys like 'state_dict', 'model', etc. If none
    is present, assumes the checkpoint itself is a state_dict.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    candidate_keys = [
        "state_dict",
        "model",
        "model_state",
        "net",
        "module",
        "model_ema",
    ]

    state_dict = None
    for key in candidate_keys:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            # Heuristic: values are tensors or nested dicts of tensors
            state_dict = checkpoint[key]
            break

    if state_dict is None:
        # Assume flat state_dict
        if isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            raise ValueError(
                f"Unsupported checkpoint structure in {checkpoint_path}. "
                "Expected a dict-like object."
            )

    return checkpoint, state_dict


def first_token(key: str) -> str:
    return key.split(".")[0]


def summarize_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for key in state_dict.keys():
        token = first_token(key)
        summary[token] = summary.get(token, 0) + 1
    return dict(sorted(summary.items(), key=lambda kv: (-kv[1], kv[0])))


def key_matches_any_prefix(key: str, prefixes: Iterable[str]) -> bool:
    for prefix in prefixes:
        if key == prefix or key.startswith(prefix + "."):
            return True
        # Also allow matching by first token
        if first_token(key) == prefix:
            return True
    return False


def merge_state_dicts(
    donor: Dict[str, torch.Tensor],
    base: Dict[str, torch.Tensor],
    replace_prefixes: Iterable[str],
    strict_missing: bool = False,
) -> Tuple[Dict[str, torch.Tensor], int, int, int]:
    """Return a merged copy of base where keys under replace_prefixes
    are replaced by those from donor when available.

    Returns (merged, num_replaced, num_missing, num_skipped).
    """
    merged = dict(base)  # shallow copy is fine; tensors are immutable here
    num_replaced = 0
    num_missing = 0
    num_skipped = 0

    replace_prefixes = list(replace_prefixes)

    for key in base.keys():
        if not key_matches_any_prefix(key, replace_prefixes):
            num_skipped += 1
            continue

        if key in donor and isinstance(donor[key], torch.Tensor):
            if donor[key].shape != base[key].shape:
                raise ValueError(
                    f"Shape mismatch for key '{key}': donor {tuple(donor[key].shape)} "
                    f"!= base {tuple(base[key].shape)}."
                )
            merged[key] = donor[key]
            num_replaced += 1
        else:
            num_missing += 1

    if strict_missing and num_missing > 0:
        raise KeyError(
            f"Missing {num_missing} keys under prefixes {replace_prefixes} in donor checkpoint."
        )

    return merged, num_replaced, num_missing, num_skipped


def save_merged(
    base_checkpoint: dict,
    merged_state_dict: Dict[str, torch.Tensor],
    output_path: str,
) -> None:
    # Prefer updating the same key used in base checkpoint if possible
    candidate_keys = [
        "state_dict",
        "model",
        "model_state",
        "net",
        "module",
        "model_ema",
    ]

    out_obj = None
    for key in candidate_keys:
        if key in base_checkpoint and isinstance(base_checkpoint[key], dict):
            out_obj = dict(base_checkpoint)
            out_obj[key] = merged_state_dict
            break

    if out_obj is None:
        # Fall back to saving the merged state_dict directly
        out_obj = merged_state_dict

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(out_obj, output_path)


def parse_prefix_list(prefix_text: str) -> Tuple[str, ...]:
    if not prefix_text:
        return tuple()
    parts = [p.strip() for p in prefix_text.split(",")]
    return tuple(p for p in parts if p)


def _maybe_reshape_for_target(donor_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """Adjust shapes for common CLIP↔timm differences if obviously safe.

    - pos_embed: (L, C) ↔ (1, L, C)
    - cls_token: (C,) or (1, C) ↔ (1, 1, C)
    """
    if donor_tensor.shape == target_tensor.shape:
        return donor_tensor

    # pos_embed: (L, C) -> (1, L, C)
    if donor_tensor.ndim == 2 and target_tensor.ndim == 3:
        if target_tensor.shape[0] == 1 and donor_tensor.shape[0] == target_tensor.shape[1] and donor_tensor.shape[1] == target_tensor.shape[2]:
            return donor_tensor.unsqueeze(0)

    # pos_embed: (1, L, C) -> (L, C)
    if donor_tensor.ndim == 3 and target_tensor.ndim == 2:
        if donor_tensor.shape[0] == 1 and donor_tensor.shape[1] == target_tensor.shape[0] and donor_tensor.shape[2] == target_tensor.shape[1]:
            return donor_tensor.squeeze(0)

    # cls_token: (C,) -> (1, 1, C)
    if donor_tensor.ndim == 1 and target_tensor.ndim == 3:
        if target_tensor.shape[0] == 1 and target_tensor.shape[1] == 1 and donor_tensor.shape[0] == target_tensor.shape[2]:
            return donor_tensor.view(1, 1, -1)

    # cls_token: (1, C) -> (1, 1, C)
    if donor_tensor.ndim == 2 and target_tensor.ndim == 3:
        if target_tensor.shape[0] == 1 and target_tensor.shape[1] == 1 and donor_tensor.shape[0] == 1 and donor_tensor.shape[1] == target_tensor.shape[2]:
            return donor_tensor.unsqueeze(1)

    # cls_token: (1, 1, C) -> (C,)
    if donor_tensor.ndim == 3 and target_tensor.ndim == 1:
        if donor_tensor.shape[0] == 1 and donor_tensor.shape[1] == 1 and donor_tensor.shape[2] == target_tensor.shape[0]:
            return donor_tensor.view(-1)

    # cls_token: (1, 1, C) -> (1, C)
    if donor_tensor.ndim == 3 and target_tensor.ndim == 2:
        if donor_tensor.shape[0] == 1 and donor_tensor.shape[1] == 1 and donor_tensor.shape[2] == target_tensor.shape[1]:
            return donor_tensor.squeeze(0)

    return donor_tensor


def detect_visual_style(state_dict: Dict[str, torch.Tensor]) -> str:
    """Detect if visual branch uses CLIP or timm style naming."""
    visual_keys = [k for k in state_dict.keys() if k.startswith("module.visual.")]
    
    # Check for CLIP-style keys
    clip_indicators = [
        "module.visual.transformer.resblocks.",
        "module.visual.class_embedding",
        "module.visual.positional_embedding",
        "module.visual.conv1.weight",
        "module.visual.ln_pre.",
    ]
    
    # Check for timm-style keys
    timm_indicators = [
        "module.visual.blocks.",
        "module.visual.cls_token",
        "module.visual.pos_embed",
        "module.visual.patch_embed.proj.weight",
        "module.visual.norm.",
    ]
    
    clip_count = sum(1 for indicator in clip_indicators if any(indicator in k for k in visual_keys))
    timm_count = sum(1 for indicator in timm_indicators if any(indicator in k for k in visual_keys))
    
    if clip_count > timm_count:
        return "clip"
    elif timm_count > clip_count:
        return "timm"
    else:
        return "unknown"


def remap_visual_bidirectional(donor: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Create a donor copy with visual keys remapped between CLIP and timm styles.

    Automatically detects donor and base styles and applies appropriate mapping.
    """
    remapped: Dict[str, torch.Tensor] = dict(donor)
    
    donor_style = detect_visual_style(donor)
    base_style = detect_visual_style(base)
    
    print(f"Detected donor visual style: {donor_style}, base visual style: {base_style}")
    
    if donor_style == base_style:
        print("Visual styles match, no remapping needed.")
        return remapped
    
    # Pre-fetch base keys for quick existence/shape checks
    base_shapes: Dict[str, torch.Size] = {k: v.shape for k, v in base.items() if isinstance(v, torch.Tensor)}

    def add_if_useful(new_key: str, tensor: torch.Tensor):
        # Only add mapping if base contains this key; otherwise leave original
        target_shape = base_shapes.get(new_key)
        if target_shape is None:
            return
        tensor_adj = _maybe_reshape_for_target(tensor, torch.empty(0).new_empty(target_shape))
        remapped[new_key] = tensor_adj

    if donor_style == "clip" and base_style == "timm":
        # CLIP → timm mapping
        print("Applying CLIP → timm visual remapping...")
        
        for key, tensor in list(donor.items()):
            if not key.startswith("module.visual."):
                continue

            # top-level simple mappings
            if key == "module.visual.class_embedding":
                add_if_useful("module.visual.cls_token", tensor)
                continue
            if key == "module.visual.positional_embedding":
                add_if_useful("module.visual.pos_embed", tensor)
                continue
            if key == "module.visual.conv1.weight":
                add_if_useful("module.visual.patch_embed.proj.weight", tensor)
                continue
            if key.startswith("module.visual.ln_post."):
                suffix = key.split(".")[-1]
                add_if_useful(f"module.visual.norm.{suffix}", tensor)
                continue
            if key == "module.visual.mask_token_embedding":
                add_if_useful("module.visual.mask_token", tensor)
                continue

            # resblocks → blocks mapping
            if key.startswith("module.visual.transformer.resblocks."):
                parts = key.split(".")
                if len(parts) >= 6:
                    idx = parts[4]
                    tail = ".".join(parts[5:])
                else:
                    continue
                
                # Attention projections
                if tail == "attn.in_proj_weight":
                    add_if_useful(f"module.visual.blocks.{idx}.attn.qkv.weight", tensor)
                    continue
                if tail == "attn.in_proj_bias":
                    add_if_useful(f"module.visual.blocks.{idx}.attn.qkv.bias", tensor)
                    continue
                if tail == "attn.out_proj.weight":
                    add_if_useful(f"module.visual.blocks.{idx}.attn.proj.weight", tensor)
                    continue
                if tail == "attn.out_proj.bias":
                    add_if_useful(f"module.visual.blocks.{idx}.attn.proj.bias", tensor)
                    continue

                # Layer norms
                if tail.startswith("ln_1."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.blocks.{idx}.norm1.{suffix}", tensor)
                    continue
                if tail.startswith("ln_2."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.blocks.{idx}.norm2.{suffix}", tensor)
                    continue

                # MLP
                if tail.startswith("mlp.c_fc."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.blocks.{idx}.mlp.fc1.{suffix}", tensor)
                    continue
                if tail.startswith("mlp.c_proj."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.blocks.{idx}.mlp.fc2.{suffix}", tensor)
                    continue

    elif donor_style == "timm" and base_style == "clip":
        # timm → CLIP mapping
        print("Applying timm → CLIP visual remapping...")
        
        for key, tensor in list(donor.items()):
            if not key.startswith("module.visual."):
                continue

            # top-level simple mappings
            if key == "module.visual.cls_token":
                add_if_useful("module.visual.class_embedding", tensor)
                continue
            if key == "module.visual.pos_embed":
                add_if_useful("module.visual.positional_embedding", tensor)
                continue
            if key == "module.visual.patch_embed.proj.weight":
                add_if_useful("module.visual.conv1.weight", tensor)
                continue
            if key.startswith("module.visual.norm."):
                suffix = key.split(".")[-1]
                add_if_useful(f"module.visual.ln_post.{suffix}", tensor)
                continue
            if key == "module.visual.mask_token":
                add_if_useful("module.visual.mask_token_embedding", tensor)
                continue

            # blocks → resblocks mapping
            if key.startswith("module.visual.blocks."):
                parts = key.split(".")
                if len(parts) >= 5:
                    idx = parts[3]
                    tail = ".".join(parts[4:])
                else:
                    continue
                
                # Attention projections
                if tail == "attn.qkv.weight":
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.attn.in_proj_weight", tensor)
                    continue
                if tail == "attn.qkv.bias":
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.attn.in_proj_bias", tensor)
                    continue
                if tail == "attn.proj.weight":
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.attn.out_proj.weight", tensor)
                    continue
                if tail == "attn.proj.bias":
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.attn.out_proj.bias", tensor)
                    continue

                # Layer norms
                if tail.startswith("norm1."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.ln_1.{suffix}", tensor)
                    continue
                if tail.startswith("norm2."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.ln_2.{suffix}", tensor)
                    continue

                # MLP
                if tail.startswith("mlp.fc1."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.mlp.c_fc.{suffix}", tensor)
                    continue
                if tail.startswith("mlp.fc2."):
                    suffix = tail.split(".")[-1]
                    add_if_useful(f"module.visual.transformer.resblocks.{idx}.mlp.c_proj.{suffix}", tensor)
                    continue

    return remapped


def categorize_missing_key(key: str) -> str:
    # Top-level
    parts = key.split(".")
    if len(parts) < 3:
        return "other"
    lvl2 = ".".join(parts[:2])  # e.g., module.visual
    # Visual (timm ViT style)
    if lvl2 == "module.visual":
        tail = ".".join(parts[2:])
        if tail.startswith("cls_token"):
            return "visual:cls_token"
        if tail.startswith("pos_embed"):
            return "visual:pos_embed"
        if tail.startswith("patch_embed."):
            return "visual:patch_embed"
        if tail.startswith("norm."):
            return "visual:norm"
        if tail.startswith("blocks."):
            # blocks.<i>.<sub>
            sub = tail.split(".", 3)
            # sub[0]=blocks, sub[1]=idx, sub[2]=attn/norm1/norm2/mlp
            if len(sub) >= 3:
                comp = sub[2]
                if comp in {"attn", "norm1", "norm2", "mlp"}:
                    return f"visual:blocks.{comp}"
            return "visual:blocks.other"
        return "visual:other"
    # Point encoder
    if lvl2 == "module.point_encoder":
        tail = ".".join(parts[2:])
        if tail.startswith("blocks.") or tail.startswith("blocks"):
            return "point_encoder:blocks"
        if tail.startswith("pos_embed"):
            return "point_encoder:pos_embed"
        for head in ["proj", "reduce_dim", "norm", "lm_head", "cls_token", "mask_token", "cls_pos"]:
            if tail.startswith(head):
                return f"point_encoder:{head}"
        return "point_encoder:other"
    # Projections / misc
    if lvl2 == "module.image_projection":
        return "image_projection"
    if lvl2 == "module.pc_projection":
        return "pc_projection"
    if lvl2 == "module.logit_scale":
        return "logit_scale"
    if lvl2 == "module.classifier":
        return "classifier"
    if lvl2 == "module.depth_encoder":
        return "depth_encoder"
    if lvl2 == "module.depth_projection":
        return "depth_projection"
    return lvl2


def analyze_missing(donor: Dict[str, torch.Tensor], base: Dict[str, torch.Tensor], prefixes: Tuple[str, ...]):
    missing = []
    replaceable = []
    shape_mismatch = []
    skipped = 0
    for key, btensor in base.items():
        if not isinstance(btensor, torch.Tensor):
            continue
        if not key_matches_any_prefix(key, prefixes):
            skipped += 1
            continue
        dtensor = donor.get(key)
        if dtensor is None:
            missing.append(key)
        else:
            if isinstance(dtensor, torch.Tensor) and dtensor.shape == btensor.shape:
                replaceable.append(key)
            else:
                shape_mismatch.append(key)
    return missing, replaceable, shape_mismatch, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge checkpoints by replacing selected module prefixes from donor into base.\n"
            "Example: replace image and point encoders while keeping depth encoder.\n"
            "Supports bidirectional visual mapping (CLIP ↔ timm)."
        )
    )
    parser.add_argument("--src-a", required=True, help="Path to donor checkpoint (.pth)")
    parser.add_argument("--src-b", required=True, help="Path to base checkpoint (.pth)")
    parser.add_argument("--out", required=True, help="Path to save merged checkpoint (.pth)")
    parser.add_argument(
        "--prefixes",
        default="",
        help=(
            "Comma-separated prefixes to replace from donor into base. "
            "Prefixes are matched as exact first token or 'prefix.' at start."
        ),
    )
    parser.add_argument(
        "--image-prefixes",
        default="",
        help="Optional comma list; convenience to include image encoder prefixes",
    )
    parser.add_argument(
        "--point-prefixes",
        default="",
        help="Optional comma list; convenience to include point/pc encoder prefixes",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Error if any target keys are missing in donor checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write output; only print summary and checks.",
    )
    parser.add_argument(
        "--auto-remap-visual",
        action="store_true",
        help=(
            "Automatically detect and remap visual branch between CLIP and timm styles "
            "(e.g., class_embedding↔cls_token, positional_embedding↔pos_embed, "
            "attn.in_proj_*↔attn.qkv.*, ln_1/ln_2↔norm1/norm2, c_fc/c_proj↔fc1/fc2)."
        ),
    )
    parser.add_argument(
        "--report-missing",
        action="store_true",
        help=(
            "Print detailed list of missing keys (present in base but absent in donor) "
            "grouped by category."
        ),
    )
    parser.add_argument(
        "--report-missing-samples",
        type=int,
        default=5,
        help="Number of example keys to show per missing category (default: 5)",
    )

    args = parser.parse_args()

    ckpt_a, sd_a = load_state_dict(args.src_a)
    ckpt_b, sd_b = load_state_dict(args.src_b)

    if args.auto_remap_visual:
        before = len([k for k in sd_a.keys() if k.startswith("module.visual.")])
        sd_a = remap_visual_bidirectional(sd_a, sd_b)
        after = len([k for k in sd_a.keys() if k.startswith("module.visual.")])
        print(f"Applied bidirectional visual remap on donor (visual keys before/after: {before}/{after}).")

    print("Donor prefixes (top-level token -> param count):")
    print(summarize_prefixes(sd_a))
    print("Base prefixes (top-level token -> param count):")
    print(summarize_prefixes(sd_b))

    user_prefixes = (
        parse_prefix_list(args.prefixes)
        + parse_prefix_list(args.image_prefixes)
        + parse_prefix_list(args.point_prefixes)
    )

    if not user_prefixes:
        raise SystemExit(
            "No prefixes provided. Use --prefixes or --image-prefixes/--point-prefixes."
        )

    print(f"Will replace prefixes: {user_prefixes}")

    merged, num_replaced, num_missing, num_skipped = merge_state_dicts(
        donor=sd_a,
        base=sd_b,
        replace_prefixes=user_prefixes,
        strict_missing=args.strict_missing,
    )

    print(
        f"Replaced: {num_replaced}, Missing: {num_missing}, Unaffected (skipped): {num_skipped}"
    )

    if args.report_missing:
        missing, replaceable, shape_mismatch, skipped = analyze_missing(sd_a, sd_b, user_prefixes)
        print(f"\n[Report] replaceable: {len(replaceable)}, missing: {len(missing)}, shape_mismatch: {len(shape_mismatch)}, skipped: {skipped}")
        # Group missing by category
        from collections import defaultdict
        groups = defaultdict(list)
        for k in missing:
            groups[categorize_missing_key(k)].append(k)
        print("[Missing grouped by category]")
        for cat, keys in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            print(f"- {cat}: {len(keys)}")
            for sample in keys[: args.report_missing_samples]:
                print(f"  * {sample}")
        if shape_mismatch:
            print("\n[Shape mismatches]")
            for sample in shape_mismatch[: args.report_missing_samples]:
                print(f"  * {sample}")

    if args.dry_run:
        print("Dry run enabled — not saving output.")
        return

    save_merged(ckpt_b, merged, args.out)
    print(f"Merged checkpoint saved to: {args.out}")


if __name__ == "__main__":
    main()
