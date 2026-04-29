#!/usr/bin/env python3
"""
Prepare single-sample NTU60 inputs from ntu60_3danno.pkl for ONNX deployment.

This script exports:
1) raw keypoint sample: (M, T, V, C)
2) model input sample:  (1, 1, M_target, T_target, V, C)
3) optional json payload for API testing

Example:
python deployment/onnx/prepare_ntu60_input_samples.py \
  --ann-file /data4/nuoyali/ProtoGCN-main/data/nturgbd/ntu60_3danno.pkl \
  --split xview_val \
  --num-samples 5 \
  --output-dir deployment/artifacts/ntu60_input_samples \
  --save-json
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import mmcv
import numpy as np
from mmcv.utils import import_modules_from_strings

from pyskl.datasets.pipelines.compose import Compose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NTU60 single-sample npy/json inputs")
    parser.add_argument(
        "--ann-file",
        default="/data4/nuoyali/ProtoGCN-main/data/nturgbd/ntu60_3danno.pkl",
        help="Path to ntu60_3danno.pkl",
    )
    parser.add_argument(
        "--split",
        default="xview_val",
        help="Split name in ann file, e.g. xview_train/xview_val/xsub_train/xsub_val",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="How many samples to export")
    parser.add_argument("--start-index", type=int, default=0, help="Start index in filtered split")
    parser.add_argument(
        "--sample-mode",
        choices=["sequential", "random", "diverse"],
        default="diverse",
        help="Sampling strategy: sequential by index / random / diverse labels first",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed used by random/diverse sampling",
    )
    parser.add_argument(
        "--output-dir",
        default="deployment/artifacts/ntu60_input_samples",
        help="Output directory",
    )
    parser.add_argument(
        "--config",
        default="configs/sem_sparse/ctrgcn_sem_sparse_ntu60_xsub_hrnet/j_aug_Xview.py",
        help="Config path used to build preprocessing pipeline",
    )
    parser.add_argument(
        "--pipeline-source",
        choices=["val", "test"],
        default="val",
        help="Choose which pipeline in config to apply",
    )
    parser.add_argument(
        "--preprocess-mode",
        choices=["pipeline", "manual"],
        default="pipeline",
        help="Use config pipeline (recommended) or manual resize-only fallback",
    )
    parser.add_argument(
        "--target-person",
        type=int,
        default=2,
        help="Target person count for model input",
    )
    parser.add_argument(
        "--target-len",
        type=int,
        default=64,
        help="Target temporal length for model input",
    )
    parser.add_argument(
        "--time-mode",
        choices=["truncate", "uniform"],
        default="truncate",
        help="How to handle T > target-len for model input",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also export json payload with keypoint list (can be large)",
    )
    return parser.parse_args()


def load_annotations(ann_file: str, split_name: Optional[str]) -> List[Dict]:
    with open(ann_file, "rb") as file:
        data = pickle.load(file)

    if isinstance(data, dict) and "annotations" in data:
        annotations = data["annotations"]
        if split_name:
            split = data.get("split", {})
            if split_name not in split:
                raise KeyError(f"Split '{split_name}' not found. Available: {list(split.keys())}")
            split_ids = set(split[split_name])
            identifier = "filename" if "filename" in annotations[0] else "frame_dir"
            annotations = [item for item in annotations if item.get(identifier) in split_ids]
        return annotations

    if isinstance(data, Sequence):
        return list(data)

    raise TypeError(f"Unsupported annotation format: {type(data)}")


def normalize_keypoint_shape(keypoint: np.ndarray) -> np.ndarray:
    keypoint = np.asarray(keypoint, dtype=np.float32)
    if keypoint.ndim == 3:
        keypoint = keypoint[None, ...]
    if keypoint.ndim != 4:
        raise ValueError(f"Expected keypoint with ndim=4, got shape={keypoint.shape}")

    if keypoint.shape[2] == 25:
        return keypoint
    if keypoint.shape[1] == 25 and keypoint.shape[2] <= 4:
        keypoint = np.transpose(keypoint, (1, 0, 2, 3))
        if keypoint.shape[2] == 25:
            return keypoint

    raise ValueError(f"Expected V=25 for NTU60, got shape={keypoint.shape}")


def resize_time_dim(keypoint: np.ndarray, target_len: int, mode: str) -> np.ndarray:
    current_len = keypoint.shape[1]
    if current_len == target_len:
        return keypoint

    if current_len > target_len:
        if mode == "truncate":
            return keypoint[:, :target_len, :, :]
        indices = np.linspace(0, current_len - 1, target_len).astype(np.int64)
        return keypoint[:, indices, :, :]

    pad = np.zeros(
        (keypoint.shape[0], target_len - current_len, keypoint.shape[2], keypoint.shape[3]),
        dtype=np.float32,
    )
    return np.concatenate([keypoint, pad], axis=1)


def resize_person_dim(keypoint: np.ndarray, target_person: int) -> np.ndarray:
    current_person = keypoint.shape[0]
    if current_person == target_person:
        return keypoint
    if current_person > target_person:
        return keypoint[:target_person, ...]

    pad = np.zeros(
        (target_person - current_person, keypoint.shape[1], keypoint.shape[2], keypoint.shape[3]),
        dtype=np.float32,
    )
    return np.concatenate([keypoint, pad], axis=0)


def make_model_input(
    keypoint: np.ndarray,
    target_person: int,
    target_len: int,
    time_mode: str,
) -> np.ndarray:
    keypoint = resize_person_dim(keypoint, target_person)
    keypoint = resize_time_dim(keypoint, target_len, time_mode)
    model_input = keypoint[None, None, ...]
    return np.ascontiguousarray(model_input, dtype=np.float32)


def sanitize_name(item: Dict, fallback_index: int) -> str:
    raw_name = item.get("frame_dir") or item.get("filename") or f"sample_{fallback_index:06d}"
    base = os.path.basename(str(raw_name)).replace("\\", "_").replace("/", "_")
    return base


def build_preprocess_pipeline(config_path: str, pipeline_source: str) -> Compose:
    cfg = mmcv.Config.fromfile(config_path)
    if cfg.get("custom_imports", None):
        import_modules_from_strings(**cfg.custom_imports)

    pipeline_cfg = cfg.val_pipeline if pipeline_source == "val" else cfg.test_pipeline
    filtered = []
    for step in pipeline_cfg:
        step_type = step.get("type", "")
        if step_type in {"Collect", "ToTensor"}:
            continue
        filtered.append(step)
    if len(filtered) == 0:
        raise RuntimeError(f"Filtered pipeline is empty from config: {config_path}")
    return Compose(filtered)


def apply_pipeline_to_item(item: Dict, pipeline: Compose) -> np.ndarray:
    results = dict(item)
    results["test_mode"] = True
    results["start_index"] = 0

    keypoint = np.asarray(results["keypoint"], dtype=np.float32)
    if keypoint.ndim == 3:
        keypoint = keypoint[None, ...]
    if keypoint.ndim != 4:
        raise ValueError(f"Expected raw keypoint ndim=4, got shape={keypoint.shape}")
    results["keypoint"] = keypoint
    results["total_frames"] = int(results.get("total_frames", keypoint.shape[1]))
    results.setdefault("num_clips", 1)

    out = pipeline(results)
    if out is None:
        raise RuntimeError("Pipeline returned None for one sample.")

    processed = out["keypoint"]
    if hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()
    else:
        processed = np.asarray(processed)

    # FormatGCNInput returns (Nc, M, T, V, C)
    if processed.ndim == 4:
        processed = processed[None, ...]
    if processed.ndim != 5:
        raise ValueError(f"Expected processed shape (Nc,M,T,V,C), got {processed.shape}")

    model_input = processed[None, ...]
    return np.ascontiguousarray(model_input, dtype=np.float32)


def select_indices(
    annotations: List[Dict],
    num_samples: int,
    start_index: int,
    mode: str,
    seed: int,
) -> List[int]:
    total = len(annotations)
    if total == 0:
        return []
    num_samples = min(max(0, num_samples), total)
    if num_samples == 0:
        return []

    if mode == "sequential":
        start = max(start_index, 0)
        end = min(start + num_samples, total)
        if start >= end:
            raise RuntimeError(f"Invalid range: start={start}, end={end}, total={total}")
        return list(range(start, end))

    rng = np.random.default_rng(seed)
    all_indices = np.arange(total, dtype=np.int64)

    if mode == "random":
        chosen = rng.choice(all_indices, size=num_samples, replace=False)
        return [int(x) for x in chosen.tolist()]

    # mode == "diverse"
    label_to_indices: Dict[int, List[int]] = {}
    unlabeled: List[int] = []
    for ann_index, item in enumerate(annotations):
        label = item.get("label", None)
        if label is None:
            unlabeled.append(ann_index)
            continue
        label = int(label)
        label_to_indices.setdefault(label, []).append(ann_index)

    selected: List[int] = []
    label_keys = sorted(label_to_indices.keys())

    # first pass: one sample per label (random within label), shuffled label order
    if label_keys:
        shuffled_labels = np.array(label_keys, dtype=np.int64)
        rng.shuffle(shuffled_labels)
        for label in shuffled_labels.tolist():
            pool = label_to_indices[int(label)]
            pick = int(rng.choice(np.array(pool, dtype=np.int64), size=1, replace=False)[0])
            selected.append(pick)
            if len(selected) >= num_samples:
                return selected[:num_samples]

    # second pass: fill remaining slots from all remaining indices
    selected_set = set(selected)
    remaining = [int(i) for i in all_indices.tolist() if int(i) not in selected_set]
    if len(remaining) > 0 and len(selected) < num_samples:
        need = num_samples - len(selected)
        extra = rng.choice(np.array(remaining, dtype=np.int64), size=need, replace=False)
        selected.extend(int(x) for x in extra.tolist())

    # fallback for heavily malformed labels
    if len(selected) < num_samples and unlabeled:
        selected_set = set(selected)
        pool = [i for i in unlabeled if i not in selected_set]
        if pool:
            need = min(num_samples - len(selected), len(pool))
            extra = rng.choice(np.array(pool, dtype=np.int64), size=need, replace=False)
            selected.extend(int(x) for x in extra.tolist())

    return selected[:num_samples]


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    annotations = load_annotations(args.ann_file, args.split)
    if len(annotations) == 0:
        raise RuntimeError("No samples found after filtering.")
    pipeline = None
    if args.preprocess_mode == "pipeline":
        pipeline = build_preprocess_pipeline(args.config, args.pipeline_source)

    selected_indices = select_indices(
        annotations=annotations,
        num_samples=args.num_samples,
        start_index=args.start_index,
        mode=args.sample_mode,
        seed=args.seed,
    )
    if len(selected_indices) == 0:
        raise RuntimeError("No sample indices selected. Check --num-samples and --start-index.")

    summary: List[Dict] = []
    selected_labels: List[int] = []

    for local_index, ann_index in enumerate(selected_indices):
        item = annotations[ann_index]
        if "keypoint" not in item:
            print(f"[Warn] Skip ann_index={ann_index}: missing 'keypoint'")
            continue

        keypoint_raw = normalize_keypoint_shape(item["keypoint"])
        if args.preprocess_mode == "pipeline":
            model_input = apply_pipeline_to_item(item, pipeline)
        else:
            model_input = make_model_input(
                keypoint=keypoint_raw,
                target_person=args.target_person,
                target_len=args.target_len,
                time_mode=args.time_mode,
            )

        sample_name = sanitize_name(item, ann_index)
        prefix = f"{local_index:03d}_{sample_name}"
        raw_npy_path = os.path.join(args.output_dir, f"{prefix}_raw.npy")
        model_npy_path = os.path.join(args.output_dir, f"{prefix}_model_input.npy")

        np.save(raw_npy_path, keypoint_raw.astype(np.float32))
        np.save(model_npy_path, model_input.astype(np.float32))

        if args.save_json:
            json_path = os.path.join(args.output_dir, f"{prefix}.json")
            payload = {
                "sample_name": sample_name,
                "label": int(item.get("label", -1)),
                "keypoint": keypoint_raw.tolist(),
                "total_frames": int(keypoint_raw.shape[1]),
                "num_person": int(keypoint_raw.shape[0]),
                "shape_m_t_v_c": list(keypoint_raw.shape),
            }
            with open(json_path, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)

        summary.append(
            {
                "ann_index": int(ann_index),
                "sample_name": sample_name,
                "label": int(item.get("label", -1)),
                "raw_shape": list(keypoint_raw.shape),
                "model_input_shape": list(model_input.shape),
                "raw_npy": raw_npy_path,
                "model_input_npy": model_npy_path,
            }
        )
        selected_labels.append(int(item.get("label", -1)))
        print(
            f"[OK] ann_index={ann_index}, sample={sample_name}, "
            f"raw_shape={tuple(keypoint_raw.shape)}, model_shape={tuple(model_input.shape)}"
        )

    unique_labels = sorted(set(selected_labels))
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "ann_file": args.ann_file,
                "split": args.split,
                "sample_mode": args.sample_mode,
                "seed": args.seed,
                "preprocess_mode": args.preprocess_mode,
                "config": args.config,
                "pipeline_source": args.pipeline_source,
                "exported": len(summary),
                "unique_labels": unique_labels,
                "num_unique_labels": len(unique_labels),
                "samples": summary,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )
    print(
        f"[Done] Exported {len(summary)} sample(s), "
        f"unique_labels={len(unique_labels)}: {unique_labels}"
    )
    print(f"[Done] Summary: {summary_path}")


if __name__ == "__main__":
    main()
