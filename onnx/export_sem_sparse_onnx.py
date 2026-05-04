#!/usr/bin/env python3
"""
Export SemSparseRecognizerGCN checkpoint to ONNX.

Design goals for Linux deployment step:
1) Do NOT depend on dataset files or ann_file paths.
2) Reuse model/config/checkpoint exactly from training project.
3) Provide a "sanity-only" mode as the first operation before ONNX export.

Example (first step: sanity check only):
python deployment/onnx/export_sem_sparse_onnx.py \
  --config configs/sem_sparse/ctrgcn_sem_sparse_ntu60_xsub_hrnet/j_aug_Xview.py \
  --checkpoint /path/to/best_epoch.pth \
  --sanity-only

Example (then export ONNX):
python deployment/onnx/export_sem_sparse_onnx.py \
  --config configs/sem_sparse/ctrgcn_sem_sparse_ntu60_xsub_hrnet/j_aug_Xview.py \
  --checkpoint /path/to/best_epoch.pth \
  --output deployment/artifacts/sem_sparse_ntu60_xview_joint.onnx \
  --opset 13
"""

import argparse
import os
from typing import Tuple

import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from pyskl.models import build_model


class SemSparseOnnxWrapper(torch.nn.Module):
    """
    A pure-Tensor forward wrapper for ONNX export.

    Why wrapper is needed:
    - Original recognizer forward_test() returns numpy at the end.
    - ONNX export requires a pure torch.Tensor computation graph.
    """

    def __init__(self, recognizer: torch.nn.Module):
        super().__init__()
        self.recognizer = recognizer

    def forward(self, keypoint: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoint: shape [B, Nc, M, T, V, C]
                     For NTU60 xview joint setup:
                     typically [1, 1, 2, 64, 25, 3].

        Returns:
            probs: shape [B, num_classes], post-softmax averaged over clips.
        """
        # Flatten batch and clips to match model's test path behavior.
        bsz, num_clips = keypoint.shape[:2]
        x = keypoint.reshape((bsz * num_clips,) + keypoint.shape[2:])

        # 1) Backbone feature extraction.
        feats = self.recognizer.extract_feat(x)

        # 2) Class-probabilities used as semantic condition for gate.
        logits_pre = self.recognizer.cls_head(feats)
        temperature = float(self.recognizer.test_cfg.get("classprob_temperature", 1.0))
        if temperature <= 0:
            temperature = 1.0
        class_probs = torch.softmax(logits_pre / temperature, dim=1)

        # 3) Semantic sparse gate.
        gate_out = self.recognizer.semantic_gate(feats, x, labels=None, class_probs=class_probs)
        mask = gate_out["mask"]
        feats_gated = feats * mask

        # 4) Residual mixing in test phase.
        alpha = float(self.recognizer.test_cfg.get("residual_alpha", 0.0))
        if alpha != 0.0:
            feats_gated = (1.0 - alpha) * feats_gated + alpha * feats

        # 5) Classification + clip averaging ("prob" mode).
        cls_score = self.recognizer.cls_head(feats_gated)
        cls_score = cls_score.reshape(bsz, num_clips, cls_score.shape[-1])
        probs = torch.softmax(cls_score, dim=2).mean(dim=1)
        return probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SemSparse model to ONNX")
    parser.add_argument("--config", required=True, help="Path to config .py")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument(
        "--output",
        default="deployment/artifacts/sem_sparse_ntu60_xview_joint.onnx",
        help="Output ONNX file path",
    )
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size B")
    parser.add_argument("--num-clips", type=int, default=1, help="Dummy clip count Nc")
    parser.add_argument("--num-person", type=int, default=2, help="Dummy person count M")
    parser.add_argument("--clip-len", type=int, default=64, help="Dummy temporal length T")
    parser.add_argument("--num-joints", type=int, default=25, help="Dummy joint count V")
    parser.add_argument("--num-channels", type=int, default=3, help="Dummy channels C")
    parser.add_argument(
        "--sanity-only",
        action="store_true",
        help="Run model build/load/forward sanity check and exit without ONNX export",
    )
    return parser.parse_args()


def build_dummy_input(
    batch_size: int,
    num_clips: int,
    num_person: int,
    clip_len: int,
    num_joints: int,
    num_channels: int,
    device: str,
) -> torch.Tensor:
    """
    Build deterministic dummy tensor for sanity check and export.
    """
    # Use zeros to reduce random numerical noise and keep logs reproducible.
    return torch.zeros(
        (batch_size, num_clips, num_person, clip_len, num_joints, num_channels),
        dtype=torch.float32,
        device=device,
    )


def run_sanity(wrapper: torch.nn.Module, dummy_input: torch.Tensor) -> Tuple[int, int]:
    """
    First-step health check:
    - forward pass succeeds
    - output shape is [B, num_classes]
    - output has finite values
    """
    with torch.no_grad():
        probs = wrapper(dummy_input)

    if probs.ndim != 2:
        raise RuntimeError(f"Unexpected output ndim={probs.ndim}, expected 2.")
    if not torch.isfinite(probs).all():
        raise RuntimeError("Sanity check failed: output contains inf/nan.")

    bsz, num_classes = probs.shape
    print(f"[Sanity] Forward success. output_shape=({bsz}, {num_classes})")
    return bsz, num_classes


def main() -> None:
    args = parse_args()

    # 1) Load config and import custom registries (semantic gate / recognizer).
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    # 2) Build recognizer from cfg.model only (no dataset construction).
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().to(args.device)

    # 3) Wrap recognizer to keep ONNX graph tensor-only.
    wrapper = SemSparseOnnxWrapper(model).eval().to(args.device)

    # 4) Build dummy input with your deployment target shape.
    dummy = build_dummy_input(
        batch_size=args.batch_size,
        num_clips=args.num_clips,
        num_person=args.num_person,
        clip_len=args.clip_len,
        num_joints=args.num_joints,
        num_channels=args.num_channels,
        device=args.device,
    )

    # ---- First operation you should run on Linux: sanity check ----
    run_sanity(wrapper, dummy)
    if args.sanity_only:
        print("[Done] Sanity-only mode finished. No ONNX file exported.")
        return

    # 5) Ensure output directory exists.
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 6) Export ONNX graph.
    torch.onnx.export(
        wrapper,
        (dummy,),
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["keypoint"],
        output_names=["probs"],
        # Keep B and Nc dynamic for flexible serving.
        dynamic_axes={
            "keypoint": {0: "batch", 1: "num_clips"},
            "probs": {0: "batch"},
        },
    )

    print(f"[Done] ONNX exported: {args.output}")
    print(f"[Info] Config: {args.config}")
    print(f"[Info] Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()

