#!/usr/bin/env python3
"""
Verify numerical consistency between PyTorch SemSparse wrapper and exported ONNX.

Example:
python deployment/onnx/verify_onnx_consistency.py \
  --config configs/sem_sparse/ctrgcn_sem_sparse_ntu60_xsub_hrnet/j_aug_Xview.py \
  --checkpoint /path/to/best_top1_acc_epoch_43.pth \
  --onnx deployment/artifacts/sem_sparse_ntu60_xview_joint.onnx
"""

import argparse
from typing import Dict

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import import_modules_from_strings

from pyskl.models import build_model
from deployment.onnx.export_sem_sparse_onnx import SemSparseOnnxWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ONNX consistency for SemSparse model")
    parser.add_argument("--config", required=True, help="Path to config .py")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--onnx", required=True, help="Path to exported ONNX file")

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="PyTorch device")
    parser.add_argument(
        "--ort-provider",
        default="cpu",
        choices=["cpu", "cuda"],
        help="ONNX Runtime execution provider",
    )

    parser.add_argument("--batch-size", type=int, default=1, help="Input batch size B")
    parser.add_argument("--num-clips", type=int, default=1, help="Input clip count Nc")
    parser.add_argument("--num-person", type=int, default=2, help="Input person count M")
    parser.add_argument("--clip-len", type=int, default=64, help="Input temporal length T")
    parser.add_argument("--num-joints", type=int, default=25, help="Input joint count V")
    parser.add_argument("--num-channels", type=int, default=3, help="Input channels C")

    parser.add_argument(
        "--input-mode",
        default="random",
        choices=["random", "zeros"],
        help="Input generation mode",
    )
    parser.add_argument("--seed", type=int, default=3407, help="Random seed for reproducibility")

    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for np.allclose")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for np.allclose")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise error when consistency check does not pass",
    )
    return parser.parse_args()


def build_input(args: argparse.Namespace, device: str) -> torch.Tensor:
    shape = (
        args.batch_size,
        args.num_clips,
        args.num_person,
        args.clip_len,
        args.num_joints,
        args.num_channels,
    )
    if args.input_mode == "zeros":
        x = torch.zeros(shape, dtype=torch.float32, device=device)
    else:
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed)
        x = torch.randn(shape, dtype=torch.float32, device=device, generator=gen)
    return x


def prepare_torch_model(args: argparse.Namespace) -> torch.nn.Module:
    cfg = mmcv.Config.fromfile(args.config)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval().to(args.device)
    wrapper = SemSparseOnnxWrapper(model).eval().to(args.device)
    return wrapper


def run_torch(wrapper: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        y = wrapper(x)
    return y.detach().cpu().numpy()


def run_onnx(onnx_path: str, x: np.ndarray, ort_provider: str) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required. Install with: pip install onnxruntime"
        ) from exc

    available = ort.get_available_providers()
    if ort_provider == "cuda":
        preferred = "CUDAExecutionProvider"
        fallback = "CPUExecutionProvider"
        if preferred in available:
            providers = [preferred, fallback]
        else:
            print("[Warn] CUDAExecutionProvider not available, fallback to CPUExecutionProvider.")
            providers = [fallback]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    y = sess.run([output_name], {input_name: x})[0]
    print(f"[Info] ONNX providers: {sess.get_providers()}")
    return y


def compute_metrics(pt: np.ndarray, ox: np.ndarray) -> Dict[str, float]:
    diff = np.abs(pt - ox)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())

    denom = np.maximum(np.abs(pt), 1e-12)
    rel = diff / denom
    max_rel_diff = float(rel.max())
    mean_rel_diff = float(rel.mean())

    pt_top1 = np.argmax(pt, axis=1)
    ox_top1 = np.argmax(ox, axis=1)
    top1_match = float((pt_top1 == ox_top1).mean())

    return {
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "top1_match_ratio": top1_match,
    }


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available for PyTorch.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    wrapper = prepare_torch_model(args)
    x_torch = build_input(args, args.device)
    x_numpy = x_torch.detach().cpu().numpy().astype(np.float32)

    y_pt = run_torch(wrapper, x_torch)
    y_ox = run_onnx(args.onnx, x_numpy, args.ort_provider)

    if y_pt.shape != y_ox.shape:
        raise RuntimeError(f"Output shape mismatch: torch={y_pt.shape}, onnx={y_ox.shape}")

    metrics = compute_metrics(y_pt, y_ox)
    passed = bool(np.allclose(y_pt, y_ox, atol=args.atol, rtol=args.rtol))

    print(f"[Info] Torch output shape: {tuple(y_pt.shape)}")
    print(f"[Info] ONNX output shape:  {tuple(y_ox.shape)}")
    print(
        "[Metrics] "
        f"max_abs_diff={metrics['max_abs_diff']:.6e}, "
        f"mean_abs_diff={metrics['mean_abs_diff']:.6e}, "
        f"max_rel_diff={metrics['max_rel_diff']:.6e}, "
        f"mean_rel_diff={metrics['mean_rel_diff']:.6e}, "
        f"top1_match={metrics['top1_match_ratio']:.3f}"
    )
    print(f"[Result] allclose(atol={args.atol}, rtol={args.rtol}) -> {passed}")

    if args.strict and not passed:
        raise RuntimeError("ONNX consistency check failed under --strict.")


if __name__ == "__main__":
    main()
