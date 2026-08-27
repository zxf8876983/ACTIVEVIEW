"""Ultralytics COCO-17 pose detector followed by VideoPose3D lifting."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ea_avs_mvp_v11.perception.pose3d_estimator import BasePose3DEstimator
from ea_avs_mvp_v11.perception.skeleton_definition import (
    SkeletonDefinition,
    get_skeleton_definition,
)


class UltralyticsPose3DEstimator(BasePose3DEstimator):
    """YOLO pose detection plus the project's existing VideoPose3D lifter."""

    def __init__(
        self,
        *,
        device: str,
        image_size: int,
        weights: Union[str, Path],
        skel_def: Optional[SkeletonDefinition] = None,
        inference_size: Optional[int] = None,
        backend_label: str = "ultralytics_yolo26n_pose",
    ) -> None:
        super().__init__(skel_def=skel_def or get_skeleton_definition(backend="h36m_17"))
        from ultralytics import YOLO

        self.device = torch.device(device)
        self.image_size = int(image_size)
        self.inference_size = int(inference_size or image_size)
        self.backend_label = str(backend_label)
        self.pose_model = YOLO(str(weights))

        videopose_dir = str(Path(__file__).resolve().parent / "videopose3d")
        if videopose_dir not in sys.path:
            sys.path.insert(0, videopose_dir)
        from common.camera import normalize_screen_coordinates
        from common.model import TemporalModel

        self._normalize_screen_coordinates = normalize_screen_coordinates
        checkpoint = Path(__file__).resolve().parent / "pretrained_h36m_detectron_coco.bin"
        self.videopose_model = TemporalModel(
            num_joints_in=17,
            in_features=2,
            num_joints_out=17,
            filter_widths=[3, 3, 3, 3, 3],
            causal=False,
            dropout=0.25,
            channels=1024,
        ).to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.videopose_model.load_state_dict(state["model_pos"])
        self.videopose_model.eval()

    @staticmethod
    def _person_score(result: object, index: int) -> float:
        boxes = result.boxes
        box_conf = float(boxes.conf[index].detach().cpu()) if boxes is not None else 0.0
        keypoints = result.keypoints
        if keypoints is None or keypoints.conf is None:
            return box_conf
        return box_conf * float(keypoints.conf[index].detach().cpu().mean())

    def _select_person(
        self,
        result: object,
        previous_box: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        keypoints = result.keypoints
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return np.zeros((17, 2), dtype=np.float32), np.zeros(17, dtype=np.float32), previous_box
        xy = keypoints.xy.detach().cpu().numpy().astype(np.float32)
        confidence = (
            keypoints.conf.detach().cpu().numpy().astype(np.float32)
            if keypoints.conf is not None
            else np.ones(xy.shape[:2], dtype=np.float32)
        )
        boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32) if result.boxes is not None else None
        scores = np.asarray(
            [self._person_score(result, index) for index in range(len(xy))],
            dtype=np.float32,
        )
        if boxes is not None and previous_box is not None:
            centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
            previous_center = (previous_box[:2] + previous_box[2:]) * 0.5
            distance = np.linalg.norm(centers - previous_center[None, :], axis=1)
            scale = max(float(np.linalg.norm(previous_box[2:] - previous_box[:2])), 1.0)
            scores *= np.exp(-distance / scale).astype(np.float32)
        index = int(np.argmax(scores))
        selected_box = boxes[index] if boxes is not None else previous_box
        return xy[index], confidence[index], selected_box

    def estimate_frame(self, rgb: np.ndarray):
        skeletons, confidence = self.estimate_sequence([rgb])
        return self._frame_result(skeletons[0], confidence[0])

    def _frame_result(self, joints: np.ndarray, confidence: np.ndarray):
        from ea_avs_mvp_v11.perception.pose3d_estimator import Pose3DEstimationResult

        return Pose3DEstimationResult(
            joints=joints,
            joint_names=self.skel_def.joint_names,
            confidence=confidence,
            coordinate_system=self.skel_def.coordinate_system.get("name", "camera_frame_right_hand"),
            estimator=f"{self.backend_label}_videopose3d",
            metadata={"backend": self.backend_label},
        )

    def estimate_sequence(self, rgb_frames: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        from common.camera import normalize_screen_coordinates

        if not rgb_frames:
            return np.zeros((0, 17, 3), dtype=np.float32), np.zeros((0, 17), dtype=np.float32)
        results = self.pose_model.predict(
            source=rgb_frames,
            imgsz=self.inference_size,
            device=str(self.device),
            verbose=False,
            conf=0.15,
            max_det=10,
        )
        points: List[np.ndarray] = []
        confidences: List[np.ndarray] = []
        previous_box: Optional[np.ndarray] = None
        for result in results:
            keypoints, confidence, previous_box = self._select_person(result, previous_box)
            points.append(keypoints)
            confidences.append(confidence)
        points_array = np.asarray(points, dtype=np.float32)
        confidence_array = np.asarray(confidences, dtype=np.float32)
        height, width = rgb_frames[0].shape[:2]
        normalized = normalize_screen_coordinates(points_array, w=width, h=height)
        padding = 121
        padded = np.concatenate(
            [
                np.repeat(normalized[:1], padding, axis=0),
                normalized,
                np.repeat(normalized[-1:], padding, axis=0),
            ],
            axis=0,
        )
        windows = [padded[index : index + 243] for index in range(len(rgb_frames))]
        predictions: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(windows), 8):
                batch = torch.from_numpy(np.asarray(windows[start : start + 8])).float().to(self.device)
                predictions.extend(self.videopose_model(batch)[:, 0].cpu().numpy())
        return np.asarray(predictions, dtype=np.float32), confidence_array
