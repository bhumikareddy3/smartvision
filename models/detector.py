"""
Object Detector
===============
Wraps YOLOv8 (Ultralytics) with ByteTrack integration.

Usage:
    detector = ObjectDetector()
    results  = detector.detect(frame)      # returns raw ultralytics Results
    tracks   = detector.track(frame)       # returns tracked Results
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np


class ObjectDetector:
    """
    YOLOv8 object detector + ByteTrack tracker.

    Parameters
    ----------
    model_name : str
        YOLOv8 variant filename, e.g. "yolov8n.pt". Downloaded automatically
        on first run and cached in the models/ directory.
    confidence : float
        Detection confidence threshold [0, 1].
    iou : float
        Non-maximum-suppression IoU threshold [0, 1].
    target_classes : list[int]
        COCO class IDs to keep. None = all classes.
    device : str
        "auto" → picks CUDA > MPS > CPU automatically.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence: float = 0.45,
        iou: float = 0.45,
        target_classes: Optional[List[int]] = None,
        device: str = "auto",
    ):
        self.confidence = confidence
        self.iou = iou
        self.target_classes = target_classes
        self.device = self._resolve_device(device)
        self.model = self._load_model(model_name)
        self._frame_count = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self, model_name: str):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        # Store models in project models/ dir to avoid re-downloading
        models_dir = Path(__file__).parent
        model_path = models_dir / model_name

        model = YOLO(str(model_path) if model_path.exists() else model_name)
        model.to(self.device)
        return model

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray):
        """Run detection WITHOUT tracking. Returns ultralytics Results object."""
        self._frame_count += 1
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.target_classes,
            verbose=False,
            stream=False,
        )
        return results[0] if results else None

    def track(self, frame: np.ndarray):
        """
        Run detection + ByteTrack tracking in a single call.

        Returns ultralytics Results object with .boxes.id populated.
        """
        self._frame_count += 1
        results = self.model.track(
            frame,
            conf=self.confidence,
            iou=self.iou,
            classes=self.target_classes,
            tracker="bytetrack.yaml",
            persist=True,          # keep track state across frames
            verbose=False,
            stream=False,
        )
        return results[0] if results else None

    def parse_tracks(self, result) -> List[dict]:
        """
        Convert a tracked Results object into a clean list of dicts.

        Each dict has keys:
          track_id, class_id, class_name, confidence,
          x1, y1, x2, y2, cx, cy
        """
        tracks = []
        if result is None or result.boxes is None:
            return tracks

        boxes = result.boxes
        ids = boxes.id  # may be None if no tracks

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            track_id = int(ids[i]) if ids is not None else -1

            tracks.append(
                dict(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=result.names.get(cls_id, str(cls_id)),
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy=cy,
                )
            )
        return tracks

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def reset(self):
        """Reset tracker state (e.g. when switching video sources)."""
        self._frame_count = 0
        # Force tracker reset by re-running a blank predict
        try:
            self.model.predictor = None
        except Exception:
            pass
