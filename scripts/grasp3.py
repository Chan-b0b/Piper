#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Autonomous grasping pipeline v3 — state-machine + background detector.

Architecture
------------
  DetectorThread  : runs GroundingDINO continuously; stores latest result.
  GraspStateMachine.step() : called once per control tick (CONTROL_DT).
    Returns pre-solved joint angles (list[float]) or None (no joint motion).
  Main loop       : quit_pressed → machine.step() → _send_joints → sleep

States
------
  SEARCH        accumulate N detections at search pose
  MOVING        generic smoothstep interpolation (shared by all motion stages)
  YAW_DETECT    wait for re-detection after yaw move
  HEIGHT_DETECT wait for detection → check convergence → move to object Z
  OPEN_GRIPPER  timed wait for gripper to open, then start approach
  GRIP          close gripper with contact feedback (non-joint ticks)
  DONE / ABORT  terminal states

Usage
-----
  python3 scripts/grasp3.py [--classes "can"] [--can can2] [--conf 0.25] [--vis]

Keys
  SPACE    : start one pick attempt
  q        : abort current attempt, return home
  Ctrl+C   : exit immediately, return home
"""

import argparse
import enum
import os
import select
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import pinocchio as pin
import pyzed.sl as sl
import torch

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

from piper_sdk import C_PiperInterface_V2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from config import GraspConfig
from utils import (
    build_fk, fk_T_base_ee, read_joints_deg,
    open_zed, sample_depth, backproject,
    read_gripper_feedback,
)
from ik_ee_ctrl import IKEndEffectorCtrl, CONTROL_DT


# ── State enum ────────────────────────────────────────────────────────────────

class State(enum.Enum):
    SEARCH        = "search"
    MOVING        = "moving"
    YAW_DETECT    = "yaw_detect"
    HEIGHT_DETECT = "height_detect"
    OPEN_GRIPPER  = "open_gripper"
    GRIP          = "grip"
    DONE          = "done"
    ABORT         = "abort"


TERMINAL_STATES = {State.DONE, State.ABORT}


# ── Geometry ──────────────────────────────────────────────────────────────────

def approach_rotation(yaw_rad: float) -> np.ndarray:
    """RPY = (0, π/2, yaw_rad) — gripper horizontal, facing yaw direction."""
    return pin.rpy.rpyToMatrix(0.0, np.pi / 2, yaw_rad)


# ── Terminal helpers ──────────────────────────────────────────────────────────

def wait_for_space() -> bool:
    """Block until Space (→ True) or q/Ctrl-C (→ False)."""
    print("\nPress SPACE to start  (q = quit)")
    orig = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())
    try:
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = os.read(sys.stdin.fileno(), 4).decode("utf-8", errors="replace")
                if ch == " ":
                    return True
                if ch in ("q", "Q", "\x03"):
                    return False
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig)


def quit_pressed() -> bool:
    """Non-blocking; raises KeyboardInterrupt on Ctrl-C."""
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = os.read(sys.stdin.fileno(), 4).decode("utf-8", errors="replace")
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch in ("q", "Q")
    return False


def rprint(msg: str):
    sys.stdout.write(msg.rstrip("\n") + "\r\n")
    sys.stdout.flush()


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_once(
    cam, rgb_mat, depth_mat, runtime, K,
    processor, det_model, device, text_prompt,
    piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
    cfg: GraspConfig,
    vis_win: str | None = None,
) -> tuple[np.ndarray, float] | None:
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
        # cy = y1 + (2.0 / 3.0) * (y2 - y1)
        cy = (y1 + y2) / 2.0
        d  = sample_depth(depth, cx, cy, cfg.depth_patch, cfg.depth_min_m, cfg.depth_max_m)
        if d is None:
            continue
        d += cfg.depth_offset_m
        p_cam  = backproject(cx, cy, d, K)
        p_base = (T_be @ T_ee_cam @ np.append(p_cam, 1.0))[:3]
        if score > best_conf:
            best_conf = score
            best_p    = p_base
        if vis_win:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if vis_win:
        annotated_frame = frame
    else:
        annotated_frame = None
    return (best_p, best_conf, annotated_frame) if best_p is not None else (None, None, annotated_frame)


class DetectorThread(threading.Thread):
    """
    Runs detect_once() in a loop; the latest result is readable at any time
    via .latest (consuming — clears after read).
    """

    def __init__(self, cam, rgb_mat, depth_mat, runtime, K,
                 processor, det_model, device, text_prompt,
                 piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
                 cfg: GraspConfig, vis_win: str | None = None):
        super().__init__(daemon=True)
        self._detect_args = (
            cam, rgb_mat, depth_mat, runtime, K,
            processor, det_model, device, text_prompt,
            piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
            cfg, vis_win,
        )
        self._lock        = threading.Lock()
        self._latest      = None   # (p_base, conf) | None
        self._latest_frame = None  # annotated BGR frame for vis (main-thread display)
        self._running     = True

    @property
    def latest(self) -> tuple[np.ndarray, float] | None:
        """Return and clear the latest detection result."""
        with self._lock:
            r, self._latest = self._latest, None
            return r

    @property
    def latest_frame(self) -> "np.ndarray | None":
        """Return (without clearing) the most recent annotated frame."""
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
                    self._latest = (p, conf)   # overwrite; consumer reads at its own pace


# ── Arm motion helpers ────────────────────────────────────────────────────────


def return_to_home(ctrl: IKEndEffectorCtrl, cfg: GraspConfig):
    print("Zeroing joint1…")
    ctrl.locked_joints = set()
    j1_zero = [round(np.rad2deg(ctrl.q[i]) * 1000) for i in range(6)]
    j1_zero[0] = 0
    ctrl.go_to_joints(j1_zero)
    print("Returning to intermediate…")
    ctrl.go_to_joints(list(cfg.intermediate_pose))
    print("Returning to zero…")
    ctrl.go_to_joints([0, 0, 0, 0, 0, 0])


# ── State machine ─────────────────────────────────────────────────────────────

class GraspStateMachine:
    """
    One call to step() per control tick.
    Returns joint angles (list[float]) when moving, or None when idle/waiting.
    """

    _GRIP_SETTLE = 0.08   # seconds between gripper close steps

    def __init__(self, ctrl: IKEndEffectorCtrl,
                 detector: DetectorThread,
                 cfg: GraspConfig):
        self.ctrl     = ctrl
        self.detector = detector
        self.cfg      = cfg
        self.state    = State.SEARCH
        self.success  = False

        # Interpolation state
        self._start_pos   = None
        self._end_pos     = None
        self._start_rot   = None
        self._end_rot     = None
        self._start_q_deg = None
        self._end_q_deg   = None
        self._step        = 0
        self._n_steps     = 0
        self._on_done     = None
        self._move_is_joints_only = False  # True = no Cartesian snap at end

        # Per-attempt data
        self._hits            = []
        self._p_obj           = None
        self._yaw_iters       = 0
        self._gripper_ready_t = 0.0
        self._grip_cmd        = 0
        self._grip_deadline   = 0.0
        self._grip_last_t     = 0.0
        self._grip_prev_angle = None   # angle at previous step command

    def reset(self):
        """Prepare for a new pick attempt."""
        self.state   = State.SEARCH
        self.success = False
        self.ctrl.locked_joints = set()
        self._hits         = []
        self._p_obj        = None
        self._yaw_iters    = 0
        self._gripper_ready_t = 0.0
        self._grip_cmd     = 0
        self._grip_deadline  = 0.0
        self._grip_last_t    = 0.0
        self._grip_prev_angle = None

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def step(self) -> list | None:
        return {
            State.SEARCH:        self._search,
            State.MOVING:        self._moving,
            State.YAW_DETECT:    self._yaw_detect,
            State.HEIGHT_DETECT: self._height_detect,
            State.OPEN_GRIPPER:  self._open_gripper_wait,
            State.GRIP:          self._grip,
        }.get(self.state, lambda: None)()

    # ── Motion primitives ─────────────────────────────────────────────────────

    def _begin_move_joints(self, target_q_deg: list, duration: float, on_done):
        """Joint-space smoothstep interpolation. No IK, no Cartesian state."""
        self._start_q_deg         = [np.rad2deg(self.ctrl.q[i]) for i in range(6)]
        self._end_q_deg           = list(target_q_deg)
        self._n_steps             = max(1, round(duration / CONTROL_DT))
        self._step                = 0
        self._on_done             = on_done
        self._move_is_joints_only = True
        self.state                = State.MOVING

    def _begin_move(self, end_pos: np.ndarray, end_rot: np.ndarray,
                    duration: float, on_done):
        self._start_pos   = self.ctrl.target_pos.copy()
        self._start_rot   = self.ctrl.target_rot.copy()
        self._end_pos     = end_pos.copy()
        self._end_rot     = end_rot.copy()
        self._start_q_deg = [np.rad2deg(self.ctrl.q[i]) for i in range(6)]

        # Pre-solve IK for the end pose so we can interpolate in joint space.
        saved_q = self.ctrl.q.copy()
        self.ctrl.target_pos = end_pos.copy()
        self.ctrl.target_rot = end_rot.copy()
        for _ in range(100):
            self.ctrl._solve_ik()
        self._end_q_deg = [np.rad2deg(self.ctrl.q[i]) for i in range(6)]
        # Restore arm state — motion has not started yet.
        self.ctrl.q          = saved_q
        self.ctrl.target_pos = self._start_pos.copy()
        self.ctrl.target_rot = self._start_rot.copy()

        self._move_is_joints_only = False
        self._n_steps = max(1, round(duration / CONTROL_DT))
        self._step    = 0
        self._on_done = on_done
        self.state    = State.MOVING

    def _moving(self) -> list:
        self._step += 1
        t = min(1.0, self._step / self._n_steps)
        t = t * t * (3.0 - 2.0 * t)   # smoothstep in joint space
        joints_deg = [
            self._start_q_deg[i] + t * (self._end_q_deg[i] - self._start_q_deg[i])
            for i in range(6)
        ]
        for i in range(6):
            self.ctrl.q[i] = np.deg2rad(joints_deg[i])
        if self._step <= 3 or self._step % 10 == 0 or self._step == self._n_steps:
            rprint(
                f"[MOV] step={self._step}/{self._n_steps}  "
                f"j1={joints_deg[0]:+.2f}°  joints_only={self._move_is_joints_only}"
            )
        if self._step >= self._n_steps:
            if not self._move_is_joints_only:
                self.ctrl.target_pos = self._end_pos.copy()
                self.ctrl.target_rot = self._end_rot.copy()
            self._on_done()
        return joints_deg

    # ── Stage 1: Search ──────────────────────────────────────────────────────

    def _search(self) -> None:
        result = self.detector.latest
        if result is not None:
            p, conf = result
            self._hits.append(p)
            sys.stdout.write(
                f"\r[S1] {len(self._hits)}/{self.cfg.detection_frames}  "
                f"conf={conf:.2f}  "
                f"X={p[0]*100:+.1f} Y={p[1]*100:+.1f} Z={p[2]*100:+.1f} cm   "
            )
            sys.stdout.flush()
            if len(self._hits) >= self.cfg.detection_frames:
                self._p_obj = np.median(np.array(self._hits), axis=0)
                self._hits.clear()
                dist = float(np.linalg.norm(self._p_obj))
                if dist > self.cfg.max_reach_m:
                    rprint(f"\n[S1] too far ({dist*100:.1f} cm) — keep looking")
                    return None
                rprint(
                    f"\n[S1] locked  "
                    f"X={self._p_obj[0]*100:+.2f} "
                    f"Y={self._p_obj[1]*100:+.2f} "
                    f"Z={self._p_obj[2]*100:+.2f} cm"
                )
                self._yaw_iters = 0
                self._begin_yaw()
        else:
            sys.stdout.write(f"\r[S1] {len(self._hits)} hits so far, waiting…   ")
            sys.stdout.flush()
        return None

    # ── Stage 2: Yaw ─────────────────────────────────────────────────────────

    def _begin_yaw(self):
        ideal_yaw_deg   = np.rad2deg(np.arctan2(self._p_obj[1], self._p_obj[0]))
        current_yaw_deg = np.rad2deg(self.ctrl.q[0])
        yaw_err         = abs(ideal_yaw_deg - current_yaw_deg)

        rprint(
            f"[S2] ideal={ideal_yaw_deg:.1f}°  "
            f"cur={current_yaw_deg:.1f}°  err={yaw_err:.1f}°"
        )

        if yaw_err <= self.cfg.yaw_threshold_deg or self._yaw_iters >= self.cfg.yaw_max_iters:
            rprint("[S2] aligned — locking joint1")
            self.ctrl.locked_joints = {0}
            self._misses = 0
            self.state   = State.HEIGHT_DETECT
            return

        target_yaw_deg = float(np.clip(ideal_yaw_deg, -154.0, 154.0))

        # Only move joint1 — keep all other joints exactly where they are.
        target_q_deg    = [np.rad2deg(self.ctrl.q[i]) for i in range(6)]
        target_q_deg[0] = target_yaw_deg

        self._yaw_q_tgt  = None   # no longer needed
        self._yaw_iters += 1

        duration = float(np.clip(abs(ideal_yaw_deg - np.rad2deg(self.ctrl.q[0])) / 45.0, 0.5, 3.0))

        def _after():
            # ctrl.q is already at target (set by _moving); sync Cartesian state.
            pin.forwardKinematics(self.ctrl.model, self.ctrl.data, self.ctrl.q)
            pin.updateFramePlacements(self.ctrl.model, self.ctrl.data)
            T = self.ctrl.data.oMf[self.ctrl.ee_id]
            self.ctrl.target_pos = T.translation.copy()
            self.ctrl.target_rot = T.rotation.copy()
            self._misses         = 0
            self._hits.clear()
            self.state = State.YAW_DETECT

        self._begin_move_joints(target_q_deg, duration, _after)

    def _yaw_detect(self) -> None:
        result = self.detector.latest
        if result is not None:
            self._p_obj  = result[0]
            self._misses = 0
            self._begin_yaw()
        else:
            self._misses += 1
            sys.stdout.write(
                f"\r[S2] waiting for re-detection (miss {self._misses})   "
            )
            sys.stdout.flush()
            if self._misses > 30:
                rprint("\n[S2] object lost — restarting search")
                self.ctrl.locked_joints = set()
                self._yaw_iters         = 0
                self._misses            = 0
                self._hits.clear()
                self.state = State.SEARCH
        return None

    # ── Stage 3: Height ──────────────────────────────────────────────────────

    def _height_detect(self) -> None:
        result = self.detector.latest
        if result is not None:
            self._p_obj  = result[0]
            self._misses = 0

            tgt_rot     = approach_rotation(self.ctrl.q[0])
            z_err       = self._p_obj[2] - self.ctrl.target_pos[2]
            dR_rem      = self.ctrl.target_rot.T @ tgt_rot
            rot_err_deg = np.rad2deg(np.linalg.norm(pin.log3(dR_rem)))

            sys.stdout.write(
                f"\r[S3] dZ={z_err*100:+.1f} cm  rot={rot_err_deg:.1f}°   "
            )
            sys.stdout.flush()

            if (abs(z_err) < self.cfg.height_threshold_m
                    and rot_err_deg < self.cfg.orient_threshold_deg):
                rprint("\n[S3] aligned — opening gripper")
                self._launch_approach()
                return None

            end_pos  = np.array([
                self.ctrl.target_pos[0],
                self.ctrl.target_pos[1],
                self._p_obj[2],
            ])
            duration = float(np.clip(abs(z_err) * 50.0, 0.5, 3.0))

            def _after():
                self._misses = 0
                self.state   = State.HEIGHT_DETECT

            self._begin_move(end_pos, tgt_rot, duration, _after)
        else:
            self._misses += 1
            sys.stdout.write(
                f"\r[S3] no detection (miss {self._misses})   "
            )
            sys.stdout.flush()
            if self._misses > 30:
                rprint("\n[S3] object lost — restarting search")
                self.ctrl.locked_joints = set()
                self._yaw_iters         = 0
                self._misses            = 0
                self._hits.clear()
                self.state = State.SEARCH
        return None

    # ── Stage 4: Approach ────────────────────────────────────────────────────

    def _launch_approach(self):
        self.ctrl.piper.GripperCtrl(
            self.ctrl.GRIPPER_OPEN_ANGLE, self.ctrl.GRIPPER_EFFORT, 0x01, 0)
        self.ctrl._gripper_open = True
        self._gripper_ready_t   = time.time() + 0.5
        self.state              = State.OPEN_GRIPPER

    def _open_gripper_wait(self) -> None:
        if time.time() < self._gripper_ready_t:
            return None

        # Rotation is already what Stage 3 converged to — keep it exactly.
        tgt_rot    = self.ctrl.target_rot.copy()
        tcp_offset = self.ctrl.get_tcp_offset_base()
        tcp_mag    = np.linalg.norm(tcp_offset)
        grasp_tcp  = np.array([self._p_obj[0], self._p_obj[1],
                                self.ctrl.target_pos[2]])
        grasp_pos  = grasp_tcp - tcp_offset

        rprint(
            f"[S4] obj  XY=({self._p_obj[0]*100:+.1f}, {self._p_obj[1]*100:+.1f}) cm"
        )
        rprint(
            f"[S4] tcp_offset magnitude={tcp_mag*100:.1f} cm  "
            f"dir=({tcp_offset[0]/tcp_mag:+.2f}, {tcp_offset[1]/tcp_mag:+.2f}, {tcp_offset[2]/tcp_mag:+.2f})"
        )
        rprint(
            f"[S4] grasp_pos XY=({grasp_pos[0]*100:+.1f}, {grasp_pos[1]*100:+.1f}) cm  "
            f"Z={grasp_pos[2]*100:+.1f} cm"
        )
        rprint("[S4] advancing to grasp position")

        def _after():
            self._grip_cmd        = self.ctrl.GRIPPER_OPEN_ANGLE
            self._grip_deadline   = time.time() + self.cfg.gripper_close_timeout_s
            self._grip_last_t     = 0.0
            self._grip_prev_angle = None
            self.state            = State.GRIP

        self._begin_move(grasp_pos, tgt_rot, self.cfg.approach_duration_s, _after)
        return None

    # ── Stage 5: Grip ────────────────────────────────────────────────────────

    def _grip(self) -> None:
        fb = read_gripper_feedback(self.ctrl.piper)

        now = time.time()
        if now - self._grip_last_t >= self._GRIP_SETTLE:
            self.ctrl.piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
            self.ctrl.piper.GripperCtrl(
                self._grip_cmd, self.ctrl.GRIPPER_EFFORT, 0x01, 0)

            # Stall detection: if angle barely moved since last command,
            # the gripper hit something.  Only check after the first step
            # (so we have a previous angle to compare against).
            stalled = (
                self._grip_prev_angle is not None
                and fb["angle_raw"] <= self.cfg.gripper_max_contact_angle_raw
                and fb["angle_raw"] > self.cfg.gripper_min_hold_angle_raw
                and not fb["faulted"]
                and (self._grip_prev_angle - fb["angle_raw"])
                     < self.cfg.gripper_stall_threshold_raw
            )

            rprint(
                f"[GRIP] cmd={self._grip_cmd}  "
                f"fb_angle={fb['angle_raw']}  "
                f"delta={0 if self._grip_prev_angle is None else self._grip_prev_angle - fb['angle_raw']}  "
                f"stalled={stalled}  faulted={fb['faulted']}"
            )

            self._grip_prev_angle = fb["angle_raw"]
            self._grip_last_t = now

            if stalled:
                rprint(
                    f"[GRIP] grasped!  "
                    f"angle={fb['angle_mm']:.1f} mm  "
                    f"effort={fb['effort_nm']:+.2f} N·m"
                )
                self.ctrl._gripper_open = False
                lift_pos = self.ctrl.target_pos + np.array([0.0, 0.0, self.cfg.lift_height_m])
                self._begin_move(lift_pos, self.ctrl.target_rot.copy(), 2.0, self._after_lift)
                return None

            # Advance closing command once gripper has caught up.
            if (self._grip_cmd > self.cfg.gripper_min_hold_angle_raw
                    and fb["angle_raw"] <= self._grip_cmd + 1500):
                self._grip_cmd = max(
                    self.cfg.gripper_min_hold_angle_raw,
                    self._grip_cmd - self.cfg.gripper_close_step_raw,
                )

        if fb["faulted"]:
            self.state = State.ABORT
            return None

        if now > self._grip_deadline:
            rprint("[GRIP] timeout — no object detected")
            self.state = State.ABORT
            return None

        return None   # no joint motion during grip

    def _after_lift(self):
        rprint("[S4] lift complete")
        self.ctrl.locked_joints = set()
        self.success = True
        self.state   = State.DONE


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Piper autonomous grasp pipeline v3")
    parser.add_argument("--classes", default=None, help="Override config.classes")
    parser.add_argument("--can",     default=None, help="Override config.can_port")
    parser.add_argument("--conf",    type=float, default=None,
                        help="Override config.min_conf")
    parser.add_argument("--vis",     action="store_true",
                        help="Show detection visualisation window")
    args = parser.parse_args()

    cfg = GraspConfig()
    if args.classes: cfg.classes  = args.classes
    if args.can:     cfg.can_port = args.can
    if args.conf:    cfg.min_conf = args.conf

    class_list  = [c.strip() for c in cfg.classes.split(",")]
    text_prompt = ". ".join(class_list) + "."
    print(f"Target: {class_list}  |  prompt: {text_prompt!r}")

    if not os.path.exists(cfg.calib_path):
        print(f"ERROR: calibration file not found: {cfg.calib_path}")
        sys.exit(1)
    T_ee_cam = np.load(cfg.calib_path)

    fk_model, fk_data, fk_ee_id = build_fk()

    piper = C_PiperInterface_V2(cfg.can_port)
    piper.ConnectPort()
    time.sleep(0.2)

    ctrl = IKEndEffectorCtrl(piper, move_duration=2.0)
    ctrl.emergency_restore()
    ctrl.enable()
    ctrl.go_to_joints([0, 0, 0, 0, 0, 0])

    print("Opening ZED camera…")
    cam, rgb_mat, depth_mat, runtime, K, _ = open_zed(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        depth_min_m=cfg.depth_min_m,
        depth_max_m=cfg.depth_max_m,
    )
    print("ZED opened.")

    print(f"Loading {cfg.detection_model}…")
    processor = AutoProcessor.from_pretrained(cfg.detection_model)
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.detection_model)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    det_model.to(device)
    print(f"Detection model ready on {device}.")

    vis_win  = "Grasp" if args.vis else None
    detector = DetectorThread(
        cam, rgb_mat, depth_mat, runtime, K,
        processor, det_model, device, text_prompt,
        piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
        cfg, vis_win,
    )
    detector.start()

    machine = GraspStateMachine(ctrl, detector, cfg)

    # ── Optional display thread (keeps cv2 off the control loop) ─────────────
    if vis_win:
        _display_stop = threading.Event()

        def _display_worker():
            while not _display_stop.is_set():
                frame = detector.latest_frame
                if frame is not None:
                    cv2.imshow(vis_win, frame)
                cv2.waitKey(1)   # blocks here — isolated from the control loop

        _display_thread = threading.Thread(target=_display_worker, daemon=True)
        _display_thread.start()
    else:
        _display_stop  = None
        _display_thread = None

    try:
        while True:
            if not wait_for_space():
                break

            print("\nMoving to search pose…")
            ctrl.go_to_joints(list(cfg.search_pose))
            print("At search pose.\n")

            machine.reset()

            orig = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            try:
                while machine.state not in TERMINAL_STATES:
                    if quit_pressed():
                        rprint("\nAborted.")
                        break
                    joints = machine.step()
                    if joints is not None:
                        ctrl._send_joints(joints)
                    time.sleep(CONTROL_DT)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig)

            if machine.success:
                print("\nGrasp succeeded!")
            else:
                print("\nGrasp failed or aborted.")
            return_to_home(ctrl, cfg)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        if _display_stop is not None:
            _display_stop.set()
        detector.stop()
        cam.close()
        if args.vis:
            cv2.destroyAllWindows()
        print("Returning home before emergency stop…")
        try:
            return_to_home(ctrl, cfg)
        except Exception:
            pass
        print("Emergency stop.")
        ctrl.emergency_stop()


if __name__ == "__main__":
    main()
