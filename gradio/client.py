import os
from typing import Dict, List, Tuple

import requests


API_BASE = os.getenv("SEM_SPARSE_API_BASE", "http://127.0.0.1:8000")
ANALYZE_VIDEO_DEMO_ENDPOINT = f"{API_BASE.rstrip('/')}/analyze/video-demo"
QA_ENDPOINT = f"{API_BASE.rstrip('/')}/qa"


def _format_topk(prediction: Dict) -> List[List]:
    rows: List[List] = []
    topk = prediction.get("topk", [])
    for i, item in enumerate(topk, start=1):
        rows.append(
            [
                i,
                item.get("label"),
                item.get("name"),
                round(float(item.get("score", 0.0)), 6),
            ]
        )
    return rows


def analyze_video(
    video_path: str,
    topk: int = 5,
    language: str = "zh",
    timeout: int = 60,
) -> Tuple[str, str, List[List], str, str, Dict, Dict]:
    if not video_path:
        raise ValueError("video_path is empty.")
    if int(topk) < 1:
        raise ValueError("topk must be >= 1.")

    with open(video_path, "rb") as file_handle:
        files = {"file": (os.path.basename(video_path), file_handle, "video/mp4")}
        data = {"topk": str(int(topk)), "language": language}
        response = requests.post(
            ANALYZE_VIDEO_DEMO_ENDPOINT,
            files=files,
            data=data,
            timeout=timeout,
        )
    response.raise_for_status()
    payload = response.json()

    predictions = payload.get("predictions", [])
    if not predictions:
        raise RuntimeError("No predictions in API response.")
    pred = predictions[0]

    mapping = payload.get("mapping", {})
    mapping_text = (
        f"video: {mapping.get('video_filename', '')}\n"
        f"map_key: {mapping.get('map_key', '')}\n"
        f"sample: {mapping.get('sample_name', '')}\n"
        f"input_type: {mapping.get('input_type', '')}\n"
        f"input_path: {mapping.get('input_path', '')}"
    )
    summary = (
        f"Top-1: {pred.get('pred_name', '')}\n"
        f"Label ID: {pred.get('pred_label', '')}\n"
        f"Score: {float(pred.get('score', 0.0)):.6f}\n"
        f"Sample: {pred.get('sample_name', '')}\n"
        f"Input Shape: {pred.get('input_shape', [])}\n"
        f"Model Input Shape: {pred.get('model_input_shape', [])}"
    )
    metrics = payload.get("metrics", {})
    if metrics:
        summary += (
            f"\nPrediction Total: {metrics.get('prediction_total_ms', '')} ms"
            f"\nAPI Total (with LLM): {metrics.get('api_total_ms', '')} ms"
        )

    table_rows = _format_topk(pred)
    explanation = payload.get("explanation", {})
    explanation_text = explanation.get("text", "")
    explanation_meta = (
        f"mode: {explanation.get('mode', '')}\n"
        f"error: {explanation.get('error', '')}"
    )
    return summary, mapping_text, table_rows, explanation_text, explanation_meta, pred, payload


def ask_question(
    prediction: Dict,
    question: str,
    language: str = "zh",
    timeout: int = 60,
) -> Tuple[str, str, Dict]:
    if not prediction:
        raise ValueError("prediction is empty. Please analyze a video first.")
    if not question or not question.strip():
        raise ValueError("question is empty.")

    response = requests.post(
        QA_ENDPOINT,
        json={
            "prediction": prediction,
            "question": question.strip(),
            "language": language,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    answer = payload.get("answer", {})
    answer_text = answer.get("text", "")
    answer_meta = (
        f"mode: {answer.get('mode', '')}\n"
        f"error: {answer.get('error', '')}"
    )
    return answer_text, answer_meta, payload
