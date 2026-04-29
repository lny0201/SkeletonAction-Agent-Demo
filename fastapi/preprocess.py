from typing import List, Optional

import numpy as np


class NTUPreprocessor:
    """Pure numpy NTU preprocessor for ONNX serving.

    This keeps Windows deployment lightweight and avoids runtime dependencies
    on MMCV/PyTorch. It approximates the inference-time path:
    `PreNormalize3D -> GenSkeFeat(['j']) -> UniformSample(64) -> FormatGCNInput(2)`.
    """

    def __init__(self, config_path: str = "", pipeline_source: str = "val"):
        self.config_path = config_path
        self.pipeline_source = pipeline_source
        self.clip_len = 64
        self.num_person = 2
        self.num_clips = 1

    @staticmethod
    def _normalize_raw_keypoint(keypoint: np.ndarray) -> np.ndarray:
        keypoint = np.asarray(keypoint, dtype=np.float32)
        if keypoint.ndim == 3:
            keypoint = keypoint[None, ...]
        if keypoint.ndim != 4:
            raise ValueError(f"Expected raw keypoint ndim=4 (M,T,V,C), got {keypoint.shape}")
        if keypoint.shape[-1] != 3:
            raise ValueError(f"Expected C=3, got {keypoint.shape}")
        if keypoint.shape[2] != 25:
            raise ValueError(f"Expected V=25 for NTU, got {keypoint.shape}")
        if keypoint.shape[0] not in (1, 2):
            raise ValueError(f"Expected M in [1,2], got {keypoint.shape[0]}")
        return keypoint

    @staticmethod
    def _unit_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm < 1e-6:
            return vector
        return vector / norm

    @classmethod
    def _angle_between(cls, v1: np.ndarray, v2: np.ndarray) -> float:
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0.0
        v1_u = cls._unit_vector(v1)
        v2_u = cls._unit_vector(v2)
        return float(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

    @staticmethod
    def _rotation_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
        if np.abs(axis).sum() < 1e-6 or abs(theta) < 1e-6:
            return np.eye(3, dtype=np.float32)
        axis = np.asarray(axis, dtype=np.float32)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
            ],
            dtype=np.float32,
        )

    def _pre_normalize_3d(self, keypoint: np.ndarray) -> np.ndarray:
        skeleton = keypoint.copy()
        num_person, total_frames, _, _ = skeleton.shape
        if np.allclose(skeleton, 0):
            return skeleton

        index0 = [i for i in range(total_frames) if not np.all(np.isclose(skeleton[0, i], 0))]
        if len(index0) == 0:
            return skeleton[:, :0]

        if num_person == 2:
            index1 = [i for i in range(total_frames) if not np.all(np.isclose(skeleton[1, i], 0))]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        if skeleton.shape[1] == 0:
            return skeleton

        center = skeleton[0, 0, 1].copy()
        mask = ((skeleton != 0).sum(-1) > 0)[..., None]
        skeleton = (skeleton - center) * mask

        joint_bottom = skeleton[0, 0, 0]
        joint_top = skeleton[0, 0, 1]
        axis_z = np.cross(joint_top - joint_bottom, [0, 0, 1])
        angle_z = self._angle_between(joint_top - joint_bottom, [0, 0, 1])
        matrix_z = self._rotation_matrix(axis_z, angle_z)
        skeleton = np.einsum("mtvc,dc->mtvd", skeleton, matrix_z)

        joint_rshoulder = skeleton[0, 0, 8]
        joint_lshoulder = skeleton[0, 0, 4]
        axis_x = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        angle_x = self._angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
        matrix_x = self._rotation_matrix(axis_x, angle_x)
        skeleton = np.einsum("mtvc,dc->mtvd", skeleton, matrix_x)

        return skeleton.astype(np.float32, copy=False)

    @staticmethod
    def _uniform_sample(keypoint: np.ndarray, clip_len: int) -> np.ndarray:
        total_frames = keypoint.shape[1]
        if total_frames <= 0:
            raise ValueError("Invalid total_frames after normalization.")
        if total_frames == clip_len:
            return keypoint
        if total_frames < clip_len:
            indices = np.mod(np.arange(clip_len), total_frames)
            return keypoint[:, indices]
        indices = np.linspace(0, total_frames - 1, clip_len).astype(np.int64)
        return keypoint[:, indices]

    @staticmethod
    def _format_gcn_input(keypoint: np.ndarray, num_person: int, num_clips: int) -> np.ndarray:
        current_person = keypoint.shape[0]
        if current_person < num_person:
            pad = np.zeros((num_person - current_person,) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate([keypoint, pad], axis=0)
        elif current_person > num_person:
            keypoint = keypoint[:num_person]

        _, total_frames, joints, channels = keypoint.shape
        if total_frames % num_clips != 0:
            raise ValueError(
                f"Expected total_frames divisible by num_clips, got T={total_frames}, Nc={num_clips}"
            )
        data = keypoint.reshape((num_person, num_clips, total_frames // num_clips, joints, channels))
        data = data.transpose(1, 0, 2, 3, 4)
        return np.ascontiguousarray(data, dtype=np.float32)

    def raw_to_model_input(
        self,
        keypoint: np.ndarray,
        total_frames: Optional[int] = None,
        num_clips: int = 1,
    ) -> np.ndarray:
        kp = self._normalize_raw_keypoint(keypoint)
        if total_frames is not None and int(total_frames) > 0:
            kp = kp[:, : int(total_frames)]
        kp = self._pre_normalize_3d(kp)
        kp = self._uniform_sample(kp, self.clip_len)
        data = self._format_gcn_input(kp, num_person=self.num_person, num_clips=num_clips)
        return np.ascontiguousarray(data[None, ...], dtype=np.float32)

    def npy_to_model_input(self, array: np.ndarray) -> np.ndarray:
        arr = np.asarray(array, dtype=np.float32)

        if arr.ndim == 6:
            return np.ascontiguousarray(arr, dtype=np.float32)

        if arr.ndim == 5:
            if arr.shape[-1] == 3 and arr.shape[-2] == 25 and arr.shape[1] in (1, 2):
                return np.ascontiguousarray(arr[None, ...], dtype=np.float32)
            converted: List[np.ndarray] = []
            for i in range(arr.shape[0]):
                converted.append(self.raw_to_model_input(arr[i]))
            return np.concatenate(converted, axis=0)

        if arr.ndim == 4:
            return self.raw_to_model_input(arr)

        raise ValueError(f"Unsupported npy shape: {arr.shape}")
