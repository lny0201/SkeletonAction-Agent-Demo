import os
from typing import Dict, List, Tuple

import numpy as np


def load_label_map(path: str) -> List[str]:
    if not path or not os.path.isfile(path):
        return []
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if text:
                labels.append(text)
    return labels


class OnnxActionRuntime:
    def __init__(self, onnx_path: str, label_map_path: str = "", provider: str = "cpu"):
        self.onnx_path = onnx_path
        self.label_map_path = label_map_path
        self.provider = provider
        self.labels = load_label_map(label_map_path)
        self.session = self._create_session()
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _create_session(self):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required. pip install onnxruntime") from exc

        available = ort.get_available_providers()
        if self.provider == "cuda":
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        return ort.InferenceSession(self.onnx_path, providers=providers)

    @property
    def providers(self) -> List[str]:
        return list(self.session.get_providers())

    @property
    def label_count(self) -> int:
        return len(self.labels)

    def predict_scores(self, model_input: np.ndarray) -> np.ndarray:
        if not isinstance(model_input, np.ndarray):
            model_input = np.asarray(model_input, dtype=np.float32)
        model_input = model_input.astype(np.float32, copy=False)
        if model_input.ndim != 6:
            raise ValueError(f"Expected input ndim=6 [B,Nc,M,T,V,C], got {model_input.shape}")
        outputs = self.session.run([self.output_name], {self.input_name: model_input})
        scores = outputs[0]
        if scores.ndim != 2:
            raise RuntimeError(f"Expected output ndim=2, got shape={scores.shape}")
        return scores.astype(np.float32, copy=False)

    def topk(self, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        k = max(1, min(int(k), int(scores.shape[1])))
        idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
        row = np.arange(scores.shape[0])[:, None]
        val = scores[row, idx]
        order = np.argsort(-val, axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        val = np.take_along_axis(val, order, axis=1)
        return idx, val

    def label_name(self, label_id: int) -> str:
        if 0 <= label_id < len(self.labels):
            return self.labels[label_id]
        return str(label_id)

    def build_prediction_item(
        self,
        idx: np.ndarray,
        val: np.ndarray,
        row: int,
        input_shape: List[int],
        model_input_shape: List[int],
        sample_name: str = "",
    ) -> Dict:
        top1 = int(idx[row, 0])
        return {
            "sample_name": sample_name,
            "pred_label": top1,
            "pred_name": self.label_name(top1),
            "score": float(val[row, 0]),
            "topk": [
                {
                    "label": int(idx[row, j]),
                    "name": self.label_name(int(idx[row, j])),
                    "score": float(val[row, j]),
                }
                for j in range(idx.shape[1])
            ],
            "input_shape": input_shape,
            "model_input_shape": model_input_shape,
        }

