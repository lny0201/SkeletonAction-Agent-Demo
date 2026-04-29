import json
import os
from typing import Dict


class VideoDemoResolveError(ValueError):
    """Raised when uploaded video cannot be mapped to an existing skeleton sample."""


class VideoDemoMapper:
    def __init__(self, mapping_path: str):
        self.mapping_path = mapping_path
        self._entries = self._load_mapping(mapping_path)

    @staticmethod
    def _normalize_key(text: str) -> str:
        return text.strip().replace("\\", "/").lower()

    def _load_mapping(self, mapping_path: str) -> Dict[str, Dict]:
        if not os.path.isfile(mapping_path):
            raise RuntimeError(f"Video mapping file not found: {mapping_path}")
        with open(mapping_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
        if not isinstance(raw, dict):
            raise RuntimeError(f"Invalid mapping file format (expect object): {mapping_path}")

        entries: Dict[str, Dict] = {}
        for map_key, cfg in raw.items():
            if not isinstance(cfg, dict):
                raise RuntimeError(f"Invalid mapping config for key={map_key}")
            if "path" not in cfg:
                raise RuntimeError(f"Missing `path` for key={map_key}")
            input_type = str(cfg.get("input_type", "npy")).strip().lower()
            if input_type not in {"npy", "npy_model_input", "json"}:
                raise RuntimeError(f"Unsupported input_type={input_type} for key={map_key}")

            normalized_key = self._normalize_key(map_key)
            entries[normalized_key] = {
                "map_key": str(map_key),
                "sample_name": str(cfg.get("sample_name", "")),
                "input_type": input_type,
                "path_config": str(cfg["path"]),
            }
        return entries

    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def unique_sample_count(self) -> int:
        names = {str(item.get("sample_name", "")).strip() for item in self._entries.values()}
        names = {x for x in names if x}
        return len(names)

    @property
    def mapped_video_count(self) -> int:
        total = 0
        for item in self._entries.values():
            key = str(item.get("map_key", "")).lower()
            if key.endswith(".mp4") or key.endswith(".avi"):
                total += 1
        return total

    def resolve(self, video_filename: str) -> Dict:
        filename = os.path.basename(video_filename or "")
        if not filename:
            raise VideoDemoResolveError("Uploaded file has empty filename.")

        stem, _ = os.path.splitext(filename)
        candidates = [
            filename,
            stem,
        ]

        matched = None
        for candidate in candidates:
            key = self._normalize_key(candidate)
            if key in self._entries:
                matched = self._entries[key]
                break
        if matched is None:
            raise VideoDemoResolveError(
                f"No skeleton sample mapping for video `{filename}`. "
                f"Current mapping entries: {self.count}"
            )

        abs_path = os.path.abspath(matched["path_config"])
        if not os.path.isfile(abs_path):
            raise VideoDemoResolveError(
                f"Mapped input file not found for video `{filename}`: {abs_path}"
            )

        return {
            "video_filename": filename,
            "map_key": matched["map_key"],
            "sample_name": matched["sample_name"],
            "input_type": matched["input_type"],
            "path": abs_path,
        }
