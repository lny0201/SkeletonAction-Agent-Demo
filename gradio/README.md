# Gradio Demo for Video Mapping Inference

This app is a thin frontend for FastAPI endpoint:

- `POST /analyze/video-demo`
- `POST /qa`

It uploads a video file, relies on filename-based mapping in:

- `deployment/artifacts/video_demo_map.json`

and displays:

- Top-1 prediction
- confidence score
- Top-K table
- mapping details
- explanation text
- follow-up QA
- full API JSON response

## 1. Install

```powershell
python -m pip install -r deployment/fastapi/requirements-fastapi.txt
python -m pip install -r deployment/gradio/requirements-gradio.txt
```

If your env already has dependency conflicts, use:

```powershell
python -m pip uninstall -y gradio gradio-client
python -m pip install "requests==2.28.2" "urllib3<1.27"
python -m pip install -r deployment/gradio/requirements-gradio.txt
python -m pip check
```

## 2. Start FastAPI

```powershell
python -m uvicorn deployment.fastapi.app:app --host 127.0.0.1 --port 8000
```

Optional env vars:

```powershell
$env:SEM_SPARSE_VIDEO_MAP="deployment/artifacts/video_demo_map.json"
$env:SEM_SPARSE_ONNX="deployment/artifacts/sem_sparse_ntu60_xview_joint.onnx"
$env:LLM_ENABLED="true"
$env:LLM_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_API_KEY="your_qwen_api_key"
$env:LLM_MODEL="qwen-plus"
```

You can also avoid environment variables by creating:

- `deployment/fastapi/llm.local.json`

Use this template:

```json
{
  "enabled": true,
  "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
  "api_key": "replace_with_your_real_api_key",
  "model": "qwen-turbo",
  "timeout": 30
}
```

This file is ignored by git and is safer than hardcoding secrets in source code.

## 3. Start Gradio

```powershell
$env:SEM_SPARSE_API_BASE="http://127.0.0.1:8000"
python -m deployment.gradio.app
```

Open:

- `http://127.0.0.1:7860`

## 4. Verify Mapping

Upload:

- `S008C001P015R001A022_cheer up.mp4`

Expected mapping:

- `deployment/artifacts/ntu60_input_samples/010_S008C001P015R001A022_model_input_cheer up.npy`

Expected behavior:

- prediction summary appears
- explanation appears
- QA box can answer follow-up questions

## 5. Common Issues

- `404 No skeleton sample mapping`: filename not in mapping JSON
- `422 Failed to load mapped sample`: mapped file path or format mismatch
- `503 Video demo mapping is not ready`: mapping file not found by backend
- `Explanation mode: template`: LLM API is disabled, missing API key, or request failed. The system falls back to template explanation.
- `pip dependency conflict (requests/openxlab/gradio)`: run

```powershell
python -m pip uninstall -y gradio gradio-client
python -m pip install "requests==2.28.2" "urllib3<1.27"
python -m pip install -r deployment/gradio/requirements-gradio.txt
```
