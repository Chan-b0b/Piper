#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Grasp pipeline configuration.

All tunable parameters live here. Override at the CLI or by editing defaults.

Usage example:
    from config import GraspConfig
    cfg = GraspConfig()
    cfg.can_port = "can1"          # programmatic override
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class GraspConfig:
    # ── Hardware ──────────────────────────────────────────────────────────────
    can_port: str   = "can2"
    calib_path: str = "/workspace/calibration/T_ee_cam.npy"

    # ── Detection model ───────────────────────────────────────────────────────
    detection_model: str = "IDEA-Research/grounding-dino-base"
    classes: str         = "Pepsi Can"       # comma-separated target class names

    # ── Detection thresholds ──────────────────────────────────────────────────
    min_conf: float        = 0.4    # minimum detection score to count
    detection_frames: int  = 3      # min queue entries needed when detections are tight
    detection_window: int  = 8     # sliding-window size for the detection queue
    detect_timeout_s: float = 10.0   # seconds to wait for a detection in S2/S3 before giving up
    detection_std_low_m: float  = 0.02   # std below this → use detection_frames (min)
    detection_std_high_m: float = 0.05   # std above this → use detection_window (max)

    # ── Depth sampling ────────────────────────────────────────────────────────
    depth_patch: int       = 5      # pixel half-side for depth median patch
    depth_min_m: float     = 0.0
    depth_max_m: float     = 3.0
    depth_offset_m: float  = 0.06   # added to detected depth (reach compensation)

    # ── Stage 1 — Sweep ───────────────────────────────────────────────────────
    scan_attempts_per_pos: int = 15  # frames without detection before sweeping
    sweep_offsets_m: List = field(default_factory=lambda: [
        [ 0.00,  0.00,  0.00],
        [ 0.00,  0.00,  0.1],
        [ 0.02,  0.03,  0.01],
        [ 0.02, -0.03, -0.01],
        [-0.02,  0.05,  0.15],
        [-0.02, -0.05,  0.15],
        [ 0.01,  0.05,  0.00],
        [ 0.01, -0.05,  0.00],
    ])

    # ── Stage 2 — Yaw alignment ───────────────────────────────────────────────
    yaw_threshold_deg: float = 5.0   # acceptable joint1 error vs ideal yaw
    yaw_max_iters: int       = 10    # max correction attempts before giving up

    # ── Stage 3 — EE height + orientation alignment ───────────────────────────
    height_threshold_m: float    = 0.005   # within this of object Z → aligned
    orient_threshold_deg: float  = 5.0    # rotation error → aligned
    align_detection_frames: int  = 3      # frames to lock height estimate
    align_z_step_m: float        = 0.005  # max Z step per IK control cycle
    align_rot_fraction: float    = 0.10   # fraction of remaining rotation per step
    align_timeout_s: float       = 15.0   # give up after this long

    # ── Stage 4 — Approach + grasp ────────────────────────────────────────────
    pregrasp_offset_m: float    = 0.02   # back off from object before closing
    approach_duration_s: float  = 3.0    # seconds for IK approach motion
    lift_height_m: float        = 0.08   # upward lift after successful grasp

    # ── Safety ────────────────────────────────────────────────────────────────
    max_reach_m: float = 0.80

    # ── Gripper ───────────────────────────────────────────────────────────────
    gripper_close_step_raw: int        = 4000
    gripper_open_step_raw: int         = 4000   # angle increment per tick when opening slowly
    gripper_contact_effort_raw: int    = 120
    gripper_min_hold_angle_raw: int    = 3000
    gripper_max_contact_angle_raw: int = 70000  # must close past this before stall counts
    gripper_stall_threshold_raw: int   = 1500   # angle delta < this = stalled on object
    gripper_squeeze_extra_raw: int     = 12000  # close this much more after contact before lifting
    gripper_close_timeout_s: float     = 10.0

    # ── Arm poses (Piper SDK units = degrees × 1000) ──────────────────────────
    search_pose: List[int] = field(
        default_factory=lambda: [0, 60000, -15000, 0, -50000, 0]
    )
    intermediate_pose: List[int] = field(
        default_factory=lambda: [0, 34196, -32149, 0, 32955, 0]
    )
    place_pose: List[int] = field(
        default_factory=lambda: [97770, 129183, -67797, -3032, 33671, 5199]
    )
