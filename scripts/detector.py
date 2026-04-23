#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Object detection helpers for the grasping pipeline.

  detect_once()   : grab one ZED frame, run GroundingDINO, return best 3-D
                    position in robot-base frame + confidence + annotated image.
  DetectorThread  : daemon thread that runs detect_once() in a loop and
                    accumulates results in a sliding-window deque.
                    .latest       → (median_p, median_conf) or None
                    .queue_size   → current fill level
                    .clear()      → discard stale data after arm moves
                    .latest_frame → most-recent annotated BGR frame for display
"""

import collections
import threading

import cv2
import numpy as np
import pyzed.sl as sl
import torch

from config import GraspConfig
from utils import read_joints_deg, fk_T_base_ee, sample_depth, backproject


# ── Single-frame detection ─────────────────────────────────────────────────────

def detect_once(
    cam, rgb_mat, depth_mat, runtime, K,
    processor, det_model, device, text_prompt,
    piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
    cfg: GraspConfig,
    vis_win: str | None = None,
) -> "tuple[np.ndarray | None, float | None, np.ndarray | None]":
    """
    Grab one frame from ZED, run GroundingDINO, return
    (best_p_base, best_conf, annotated_frame).

    best_p_base is None if no detection passed depth-sampling.
    annotated_frame is None when vis_win is None.
    Returns None (not a tuple) when cam.grab() fails.
    """
    if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return None
    cam.retrieve_image(rgb_mat, sl.VIEW.LEFT)
    cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
    frame = np.ascontiguousarray(rgb_mat.get_data()[:, :, :3])
    depth = depth_mat.get_data()

    inputs = processor(images=frame, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det_model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        target_sizes=[(frame.shape[0], frame.shape[1])],
    )[0]

    mask   = results["scores"] >= cfg.min_conf
    boxes  = results["boxes"][mask].cpu().numpy()
    scores = results["scores"][mask].cpu().numpy()
    labels = [results["text_labels"][i] for i, m in enumerate(mask) if m]

    joints = read_joints_deg(piper)
    T_be   = fk_T_base_ee(fk_model, fk_data, fk_ee_id, joints)

    best_p, best_conf = None, 0.0
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        d  = sample_depth(depth, cx, cy, cfg.depth_patch, cfg.depth_min_m, cfg.depth_max_m)
        if d is None:
            # Centroid patch invalid — fall back to the whole bounding box.
            bx0 = max(0, int(x1)); bx1 = min(depth.shape[1], int(x2) + 1)
            by0 = max(0, int(y1)); by1 = min(depth.shape[0], int(y2) + 1)
            region = depth[by0:by1, bx0:bx1].flatten()
            valid  = region[np.isfinite(region)
                            & (region > cfg.depth_min_m)
                            & (region < cfg.depth_max_m)]
            if len(valid) == 0:
                continue
            d = float(np.median(valid))
        d += cfg.depth_offset_m
        p_cam  = backproject(cx, cy, d, K)
        p_base = (T_be @ T_ee_cam @ np.append(p_cam, 1.0))[:3]
        if score > best_conf:
            best_conf = score
            best_p = p_base
        if vis_win:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    annotated_frame = frame if vis_win else None
    return (best_p, best_conf, annotated_frame)


# ── Background detector thread ─────────────────────────────────────────────────

class DetectorThread(threading.Thread):
    """
    Runs detect_once() in a tight loop in a daemon thread.

    Results are pushed into a sliding-window deque of size
    cfg.detection_window.  The caller reads a *median* over
    the window via .latest — no result is ever discarded on read.
    Call .clear() after the arm has moved to discard stale positions.
    """

    def __init__(self, cam, rgb_mat, depth_mat, runtime, K,
                 processor, det_model, device, text_prompt,
                 piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
                 cfg: GraspConfig, vis_win: str | None = None):
        super().__init__(daemon=True)
        self._static_args = (
            cam, rgb_mat, depth_mat, runtime, K,
            processor, det_model, device, text_prompt,
            piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
        )
        self._vis_win      = vis_win
        self.cfg           = cfg
        self._lock         = threading.Lock()
        self._queue        = collections.deque(maxlen=cfg.detection_window)
        self._min_count    = cfg.detection_frames
        self._latest_frame = None
        self._running      = True

    @property
    def _detect_args(self):
        return self._static_args + (self.cfg, self._vis_win)

    @property
    def latest(self) -> "tuple[np.ndarray, float] | None":
        """Return (median_p, median_conf) when _min_count consistent detections exist."""
        with self._lock:
            if len(self._queue) < self._min_count:
                return None
            pts   = np.array([p for p, c in self._queue])
            confs = [c for p, c in self._queue]
            return np.median(pts, axis=0), float(np.median(confs))

    @property
    def min_count(self) -> int:
        """Current adaptive required frame count."""
        return self._min_count

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self):
        """Discard all queued detections (call after the arm moves)."""
        with self._lock:
            self._queue.clear()

    @property
    def latest_frame(self) -> "np.ndarray | None":
        """Most-recent annotated BGR frame (not cleared on read)."""
        with self._lock:
            return self._latest_frame

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            result = detect_once(*self._detect_args)
            if result is None:
                continue
            p, conf, frame = result
            with self._lock:
                if frame is not None:
                    self._latest_frame = frame
                if p is not None:
                    self._queue.append((p, conf))
                    # Recompute adaptive required frame count from current queue std.
                    if len(self._queue) >= 2:
                        pts   = np.array([pt for pt, c in self._queue])
                        std_m = float(np.mean(np.std(pts, axis=0)))
                        lo    = self.cfg.detection_std_low_m
                        hi    = self.cfg.detection_std_high_m
                        t     = float(np.clip((std_m - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
                        self._min_count = round(
                            self.cfg.detection_frames
                            + t * (self.cfg.detection_window - self.cfg.detection_frames)
                        )
                    else:
                        self._min_count = self.cfg.detection_frames
