import json
import os
from pathlib import Path
from typing import Dict, List

import requests

from deployment.fastapi.prompt_builder import (
    build_explanation_messages,
    build_qa_messages,
    template_answer,
    template_explain,
)


LOCAL_CONFIG_PATH = Path(__file__).with_name("llm.local.json")
DEFAULT_LLM_CONFIG = {
    "enabled": True,
    "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "api_key": "",
    "model": "qwen-turbo",
    "timeout": 30.0,
}


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def load_llm_config() -> Dict:
    config = dict(DEFAULT_LLM_CONFIG)

    if LOCAL_CONFIG_PATH.is_file():
        with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as file_handle:
            local_cfg = json.load(file_handle)
        if isinstance(local_cfg, dict):
            config.update(local_cfg)

    if "LLM_ENABLED" in os.environ:
        config["enabled"] = _parse_bool(os.environ["LLM_ENABLED"])
    if "LLM_API_BASE" in os.environ:
        config["api_base"] = os.environ["LLM_API_BASE"]
    if "LLM_API_KEY" in os.environ:
        config["api_key"] = os.environ["LLM_API_KEY"]
    if "LLM_MODEL" in os.environ:
        config["model"] = os.environ["LLM_MODEL"]
    if "LLM_TIMEOUT" in os.environ:
        config["timeout"] = float(os.environ["LLM_TIMEOUT"])

    config["enabled"] = _parse_bool(config.get("enabled", True))
    config["api_base"] = str(config.get("api_base", "")).strip()
    config["api_key"] = str(config.get("api_key", "")).strip()
    config["model"] = str(config.get("model", "")).strip()
    config["timeout"] = float(config.get("timeout", 30.0))
    return config


def inspect_local_config() -> Dict:
    status = {
        "path": str(LOCAL_CONFIG_PATH),
        "exists": LOCAL_CONFIG_PATH.is_file(),
        "loaded": False,
        "error": "",
    }
    if not status["exists"]:
        return status
    try:
        with open(LOCAL_CONFIG_PATH, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if not isinstance(payload, dict):
            raise ValueError("llm.local.json must be a JSON object.")
        status["loaded"] = True
        return status
    except Exception as exc:
        status["error"] = str(exc)
        return status


class LLMExplainer:
    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "",
        timeout: float = 0.0,
        enabled: bool = True,
    ):
        local_status = inspect_local_config()
        cfg = load_llm_config()
        self.api_base = (api_base or cfg["api_base"]).rstrip("/")
        self.api_key = (api_key or cfg["api_key"]).strip()
        self.model = (model or cfg["model"]).strip()
        self.timeout = float(timeout or cfg["timeout"])
        self.enabled = bool(enabled if api_base or api_key or model or timeout else cfg["enabled"])
        self.local_config_path = local_status["path"]
        self.local_config_exists = bool(local_status["exists"])
        self.local_config_loaded = bool(local_status["loaded"])
        self.local_config_error = str(local_status["error"])

    @property
    def ready(self) -> bool:
        return self.enabled and bool(self.api_key) and bool(self.model) and bool(self.api_base)

    @property
    def config_path(self) -> str:
        return str(LOCAL_CONFIG_PATH)

    def _chat_completions(self, messages: List[Dict[str, str]]) -> str:
        if not self.ready:
            raise RuntimeError("LLM API is not configured or disabled.")
        response = requests.post(
            f"{self.api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.3,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("LLM response does not contain choices.")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            content = "\n".join(parts)
        content = str(content).strip()
        if not content:
            raise RuntimeError("LLM response content is empty.")
        return content

    def explain(self, prediction: Dict, language: str = "zh", include_prompt_preview: bool = False) -> Dict:
        messages = build_explanation_messages(prediction=prediction, language=language)
        fallback = template_explain(prediction=prediction, language=language)
        result = {
            "mode": "template",
            "text": fallback,
            "error": "",
            "prompt_preview": messages[-1]["content"] if include_prompt_preview else "",
        }
        try:
            text = self._chat_completions(messages)
            result["mode"] = "llm"
            result["text"] = text
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result

    def answer_question(
        self,
        prediction: Dict,
        question: str,
        language: str = "zh",
        include_prompt_preview: bool = False,
    ) -> Dict:
        messages = build_qa_messages(prediction=prediction, question=question, language=language)
        fallback = template_answer(prediction=prediction, question=question, language=language)
        result = {
            "mode": "template",
            "text": fallback,
            "error": "",
            "prompt_preview": messages[-1]["content"] if include_prompt_preview else "",
        }
        try:
            text = self._chat_completions(messages)
            result["mode"] = "llm"
            result["text"] = text
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result
