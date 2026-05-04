import io
import json
import os
import time
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from deployment.fastapi.llm import LLMExplainer
from deployment.fastapi.preprocess import NTUPreprocessor
from deployment.fastapi.runtime import OnnxActionRuntime
from deployment.fastapi.schemas import (
    AnalyzeVideoDemoResponse,
    ExplainRequest,
    ExplainResponse,
    PredictJsonRequest,
    PredictResponse,
    PredictVideoDemoResponse,
    QARequest,
    QAResponse,
)
from deployment.fastapi.video_mapper import VideoDemoMapper, VideoDemoResolveError


ONNX_PATH = os.getenv("SEM_SPARSE_ONNX", "deployment/artifacts/sem_sparse_ntu60_xview_joint.onnx")
LABEL_MAP_PATH = os.getenv("SEM_SPARSE_LABEL_MAP", "tools/data/label_map/nturgbd_120.txt")
CONFIG_PATH = os.getenv(
    "SEM_SPARSE_CONFIG",
    "configs/sem_sparse/ctrgcn_sem_sparse_ntu60_xsub_hrnet/j_aug_Xview.py",
)
PIPELINE_SOURCE = os.getenv("SEM_SPARSE_PIPELINE", "val")
ORT_PROVIDER = os.getenv("SEM_SPARSE_ORT_PROVIDER", "cpu")
PREPROCESS_BACKEND = os.getenv("SEM_SPARSE_PREPROCESS_BACKEND", "numpy")
VIDEO_DEMO_MAP_PATH = os.getenv("SEM_SPARSE_VIDEO_MAP", "deployment/artifacts/video_demo_map.json")


app = FastAPI(title="SemSparse ONNX API", version="0.1.0")
runtime: OnnxActionRuntime = None
preprocessor: NTUPreprocessor = None
video_demo_mapper: VideoDemoMapper = None
llm_explainer: LLMExplainer = None
onnx_size_mb: float = 0.0


@app.on_event("startup")
def startup_event() -> None:
    global runtime, preprocessor, video_demo_mapper, llm_explainer, onnx_size_mb
    if not os.path.isfile(ONNX_PATH):
        raise RuntimeError(f"ONNX file not found: {ONNX_PATH}")
    runtime = OnnxActionRuntime(
        onnx_path=ONNX_PATH,
        label_map_path=LABEL_MAP_PATH,
        provider=ORT_PROVIDER,
    )
    preprocessor = NTUPreprocessor(
        config_path=CONFIG_PATH,
        pipeline_source=PIPELINE_SOURCE,
    )
    if os.path.isfile(VIDEO_DEMO_MAP_PATH):
        video_demo_mapper = VideoDemoMapper(mapping_path=VIDEO_DEMO_MAP_PATH)
    else:
        video_demo_mapper = None
    llm_explainer = LLMExplainer()
    onnx_size_mb = round(os.path.getsize(ONNX_PATH) / (1024 * 1024), 3)


@app.get("/health")
def health() -> dict:
    ready = runtime is not None and preprocessor is not None
    return {
        "ready": bool(ready),
        "onnx": ONNX_PATH,
        "onnx_size_mb": onnx_size_mb,
        "providers": runtime.providers if ready else [],
        "label_count": runtime.label_count if ready else 0,
        "config": CONFIG_PATH,
        "pipeline_source": PIPELINE_SOURCE,
        "preprocess_backend": PREPROCESS_BACKEND,
        "video_demo_map": VIDEO_DEMO_MAP_PATH,
        "video_demo_map_ready": bool(video_demo_mapper is not None),
        "video_demo_map_count": video_demo_mapper.count if video_demo_mapper is not None else 0,
        "video_demo_unique_samples": video_demo_mapper.unique_sample_count if video_demo_mapper is not None else 0,
        "video_demo_mapped_videos": video_demo_mapper.mapped_video_count if video_demo_mapper is not None else 0,
        "llm_enabled": llm_explainer.enabled if llm_explainer is not None else False,
        "llm_ready": llm_explainer.ready if llm_explainer is not None else False,
        "llm_api_base": llm_explainer.api_base if llm_explainer is not None else "",
        "llm_model": llm_explainer.model if llm_explainer is not None else "",
        "llm_local_config_path": llm_explainer.local_config_path if llm_explainer is not None else "",
        "llm_local_config_exists": llm_explainer.local_config_exists if llm_explainer is not None else False,
        "llm_local_config_loaded": llm_explainer.local_config_loaded if llm_explainer is not None else False,
        "llm_local_config_error": llm_explainer.local_config_error if llm_explainer is not None else "",
    }


def _predict_from_model_input(
    model_input: np.ndarray,
    topk: int,
    sample_names: List[str],
    input_shapes: List[List[int]],
    include_metrics: bool = False,
):
    infer_t0 = time.perf_counter()
    scores = runtime.predict_scores(model_input)
    idx, val = runtime.topk(scores, topk)
    infer_ms = round((time.perf_counter() - infer_t0) * 1000, 3)
    items = []
    for i in range(model_input.shape[0]):
        sample_name = sample_names[i] if i < len(sample_names) else ""
        input_shape = input_shapes[i] if i < len(input_shapes) else list(model_input[i].shape)
        items.append(
            runtime.build_prediction_item(
                idx=idx,
                val=val,
                row=i,
                input_shape=input_shape,
                model_input_shape=list(model_input[i].shape),
                sample_name=sample_name,
            )
        )
    payload = {"num_samples": len(items), "predictions": items}
    if include_metrics:
        payload["metrics"] = {
            "onnx_inference_ms": infer_ms,
            "batch_size": int(model_input.shape[0]),
        }
    return payload


def _resolve_video_demo_prediction(filename: str, topk: int) -> Dict:
    t_total = time.perf_counter()
    if topk < 1:
        raise HTTPException(status_code=400, detail="topk must be >= 1")
    if video_demo_mapper is None:
        raise HTTPException(
            status_code=503,
            detail=f"Video demo mapping is not ready. Expected mapping file: {VIDEO_DEMO_MAP_PATH}",
        )
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded video filename is empty")

    t_map = time.perf_counter()
    try:
        mapped = video_demo_mapper.resolve(filename)
    except VideoDemoResolveError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    map_ms = round((time.perf_counter() - t_map) * 1000, 3)

    input_type = mapped["input_type"]
    sample_name = mapped["sample_name"] or mapped["video_filename"]
    input_shapes: List[List[int]]

    t_load = time.perf_counter()
    try:
        if input_type in {"npy", "npy_model_input"}:
            arr = np.load(mapped["path"], allow_pickle=False)
            if input_type == "npy_model_input":
                model_input = np.asarray(arr, dtype=np.float32)
                if model_input.ndim == 5:
                    model_input = model_input[None, ...]
                if model_input.ndim != 6:
                    raise ValueError(
                        f"npy_model_input expects ndim=6 [B,Nc,M,T,V,C], got shape={model_input.shape}"
                    )
                model_input = np.ascontiguousarray(model_input, dtype=np.float32)
            else:
                model_input = preprocessor.npy_to_model_input(arr)

            if model_input.shape[0] == 1:
                input_shapes = [list(arr.shape)]
            else:
                if arr.ndim == 5:
                    input_shapes = [list(arr[i].shape) for i in range(arr.shape[0])]
                else:
                    input_shapes = [list(arr.shape)] * model_input.shape[0]
        elif input_type == "json":
            with open(mapped["path"], "r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            if "keypoint" not in payload:
                raise ValueError("Mapped json does not contain `keypoint` field")
            raw = np.asarray(payload["keypoint"], dtype=np.float32)
            model_input = preprocessor.raw_to_model_input(
                keypoint=raw,
                total_frames=payload.get("total_frames"),
                num_clips=1,
            )
            input_shapes = [list(raw.shape)]
        else:
            raise ValueError(f"Unsupported mapped input_type: {input_type}")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to load mapped sample: {exc}") from exc
    load_preprocess_ms = round((time.perf_counter() - t_load) * 1000, 3)

    sample_names = [sample_name for _ in range(model_input.shape[0])]
    result = _predict_from_model_input(
        model_input=model_input,
        topk=topk,
        sample_names=sample_names,
        input_shapes=input_shapes,
        include_metrics=True,
    )
    metrics = dict(result.get("metrics", {}))
    metrics["mapping_lookup_ms"] = map_ms
    metrics["sample_load_preprocess_ms"] = load_preprocess_ms
    metrics["prediction_total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)
    result["metrics"] = metrics
    return {
        "mapping": {
            "video_filename": mapped["video_filename"],
            "map_key": mapped["map_key"],
            "sample_name": sample_name,
            "input_type": input_type,
            "input_path": mapped["path"],
        },
        **result,
    }


@app.post("/predict/npy", response_model=PredictResponse)
async def predict_npy(file: UploadFile = File(...), topk: int = Form(5)) -> dict:
    if topk < 1:
        raise HTTPException(status_code=400, detail="topk must be >= 1")
    try:
        payload = await file.read()
        arr = np.load(io.BytesIO(payload), allow_pickle=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load npy file: {exc}") from exc

    try:
        model_input = preprocessor.npy_to_model_input(arr)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid input shape/content: {exc}") from exc

    if model_input.shape[0] == 1:
        input_shapes = [list(arr.shape)]
    else:
        input_shapes = [list(arr[i].shape) for i in range(arr.shape[0])] if arr.ndim == 5 else [list(arr.shape)] * model_input.shape[0]
    sample_names = [file.filename or "" for _ in range(model_input.shape[0])]
    return _predict_from_model_input(model_input=model_input, topk=topk, sample_names=sample_names, input_shapes=input_shapes)


@app.post("/predict/json", response_model=PredictResponse)
def predict_json(req: PredictJsonRequest) -> dict:
    if len(req.samples) == 0:
        raise HTTPException(status_code=400, detail="samples must not be empty")

    model_inputs = []
    input_shapes: List[List[int]] = []
    sample_names: List[str] = []
    for sample in req.samples:
        try:
            raw = np.asarray(sample.keypoint, dtype=np.float32)
            model_input = preprocessor.raw_to_model_input(
                keypoint=raw,
                total_frames=sample.total_frames,
                num_clips=1,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Invalid JSON sample ({sample.sample_name}): {exc}") from exc
        model_inputs.append(model_input)
        input_shapes.append(list(raw.shape))
        sample_names.append(sample.sample_name)

    batch_input = np.concatenate(model_inputs, axis=0)
    return _predict_from_model_input(
        model_input=batch_input,
        topk=req.topk,
        sample_names=sample_names,
        input_shapes=input_shapes,
    )


@app.post("/predict/video-demo", response_model=PredictVideoDemoResponse)
async def predict_video_demo(file: UploadFile = File(...), topk: int = Form(5)) -> dict:
    filename = file.filename or ""
    return _resolve_video_demo_prediction(filename=filename, topk=topk)


@app.post("/explain", response_model=ExplainResponse)
def explain_prediction(req: ExplainRequest) -> dict:
    explanation = llm_explainer.explain(
        prediction=req.prediction.dict(),
        language=req.language,
        include_prompt_preview=req.include_prompt_preview,
    )
    return {"explanation": explanation}


@app.post("/qa", response_model=QAResponse)
def answer_question(req: QARequest) -> dict:
    answer = llm_explainer.answer_question(
        prediction=req.prediction.dict(),
        question=req.question,
        language=req.language,
        include_prompt_preview=req.include_prompt_preview,
    )
    return {"answer": answer}


@app.post("/analyze/video-demo", response_model=AnalyzeVideoDemoResponse)
async def analyze_video_demo(file: UploadFile = File(...), topk: int = Form(5), language: str = Form("zh")) -> dict:
    t_total = time.perf_counter()
    result = _resolve_video_demo_prediction(filename=file.filename or "", topk=topk)
    predictions = result.get("predictions", [])
    if not predictions:
        raise HTTPException(status_code=500, detail="Prediction result is empty.")
    t_llm = time.perf_counter()
    explanation = llm_explainer.explain(prediction=predictions[0], language=language)
    llm_ms = round((time.perf_counter() - t_llm) * 1000, 3)
    metrics = dict(result.get("metrics", {}))
    metrics["llm_explanation_ms"] = llm_ms
    metrics["api_total_ms"] = round((time.perf_counter() - t_total) * 1000, 3)
    return {
        **result,
        "explanation": explanation,
        "metrics": metrics,
    }
