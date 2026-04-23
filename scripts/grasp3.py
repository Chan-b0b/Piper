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
import collections
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
from detector import DetectorThread
from utils import build_fk, open_zed, read_gripper_feedback
from ik_ee_ctrl import IKEndEffectorCtrl, CONTROL_DT


# ── State enum ────────────────────────────────────────────────────────────────

class State(enum.Enum):
    SEARCH        = "search"
    MOVING        = "moving"
    YAW_DETECT    = "yaw_detect"
    HEIGHT_DETECT = "height_detect"
    OPEN_GRIPPER  = "open_gripper"
    GRIP          = "grip"
    PLACE         = "place"
    RELEASE       = "release"
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


# ── Arm motion helpers ────────────────────────────────────────────────────────


def return_to_home(ctrl: IKEndEffectorCtrl, cfg: GraspConfig):
    print("Zeroing joint1…")
    ctrl.locked_joints = set()
    # j1_zero = [round(np.rad2deg(ctrl.q[i]) * 1000) for i in range(6)]
    # j1_zero[0] = 0
    # ctrl.go_to_joints(j1_zero)
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
        self._p_obj           = None
        self._yaw_iters       = 0
        self._detect_deadline = 0.0
        # Sweep state
        self._sweep_idx      = 0
        self._sweep_base_pos = None
        self._sweep_base_rot = None
        self._sweep_deadline = 0.0
        self._gripper_ready_t  = 0.0
        self._grip_cmd         = 0
        self._grip_deadline    = 0.0
        self._grip_last_t      = 0.0
        self._grip_prev_angle  = None   # angle at previous step command
        self._gripper_open_cmd = self.ctrl.GRIPPER_OPEN_ANGLE

    def reset(self):
        """Prepare for a new pick attempt."""
        self.state   = State.SEARCH
        self.success = False
        self.ctrl.locked_joints = set()
        self.detector.clear()
        self._p_obj        = None
        self._yaw_iters    = 0
        self._detect_deadline = 0.0
        # Capture the current arm pose as sweep base (arm is at search pose when reset() is called)
        pin.forwardKinematics(self.ctrl.model, self.ctrl.data, self.ctrl.q)
        pin.updateFramePlacements(self.ctrl.model, self.ctrl.data)
        _T = self.ctrl.data.oMf[self.ctrl.ee_id]
        self._sweep_base_pos     = _T.translation.copy()
        self._sweep_base_rot     = _T.rotation.copy()
        self.ctrl.target_pos     = _T.translation.copy()
        self.ctrl.target_rot     = _T.rotation.copy()
        self._sweep_idx          = 0
        self._sweep_deadline     = time.time() + self.cfg.detect_timeout_s
        self._gripper_ready_t = 0.0
        self._grip_cmd     = 0
        self._grip_deadline  = 0.0
        self._grip_last_t    = 0.0
        self._grip_prev_angle = None
        self._gripper_open_cmd = self.ctrl.GRIPPER_OPEN_ANGLE

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def step(self) -> list | None:
        return {
            State.SEARCH:        self._search,
            State.MOVING:        self._moving,
            State.YAW_DETECT:    self._yaw_detect,
            State.HEIGHT_DETECT: self._height_detect,
            State.OPEN_GRIPPER:  self._open_gripper_wait,
            State.GRIP:          self._grip,
            State.PLACE:         lambda: None,
            State.RELEASE:       lambda: None,
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
        if self._step >= self._n_steps:
            if not self._move_is_joints_only:
                self.ctrl.target_pos = self._end_pos.copy()
                self.ctrl.target_rot = self._end_rot.copy()
            self._on_done()
        return joints_deg

    # ── Stage 1: Search ──────────────────────────────────────────────────────

    def _search(self) -> None:
        n = self.detector.queue_size
        result = self.detector.latest
        if result is not None:
            p, conf = result
            sys.stdout.write(
                f"\r[S1] {n}/{self.cfg.detection_window}  "
                f"conf={conf:.2f}  "
                f"X={p[0]*100:+.1f} Y={p[1]*100:+.1f} Z={p[2]*100:+.1f} cm   "
            )
            sys.stdout.flush()
            dist = float(np.linalg.norm(p))
            if dist > self.cfg.max_reach_m:
                rprint(f"\n[S1] too far ({dist*100:.1f} cm) — keep looking")
                return None
            self._p_obj = p
            rprint(
                f"\n[S1] locked  "
                f"X={p[0]*100:+.2f} "
                f"Y={p[1]*100:+.2f} "
                f"Z={p[2]*100:+.2f} cm"
            )
            self._yaw_iters = 0
            self._begin_yaw()
        elif time.time() > self._sweep_deadline:
            # No detection at current position — try next sweep offset.
            self._sweep_idx = (self._sweep_idx + 1) % len(self.cfg.sweep_offsets_m)
            offset = np.array(self.cfg.sweep_offsets_m[self._sweep_idx])
            sweep_pos = self._sweep_base_pos + offset
            rprint(
                f"\n[S1] no detection — sweep {self._sweep_idx}  "
                f"({offset[0]*100:+.1f}, {offset[1]*100:+.1f}, {offset[2]*100:+.1f}) cm"
            )
            def _after_sweep():
                self.detector.clear()
                self._sweep_deadline = time.time() + self.cfg.detect_timeout_s
                self.state = State.SEARCH
            self._begin_move(sweep_pos, self._sweep_base_rot, 1.5, _after_sweep)
        else:
            remaining = self._sweep_deadline - time.time()
            sys.stdout.write(
                f"\r[S1] {n}/{self.detector.min_count} (need {self.detector.min_count}) waiting… ({remaining:.0f}s)   "
            )
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
            self._detect_deadline = time.time() + self.cfg.detect_timeout_s
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
            self.detector.clear()
            self._detect_deadline = time.time() + self.cfg.detect_timeout_s
            self.state = State.YAW_DETECT

        self._begin_move_joints(target_q_deg, duration, _after)

    def _yaw_detect(self) -> None:
        result = self.detector.latest
        if result is not None:
            self._p_obj = result[0]
            self._begin_yaw()
        else:
            remaining = self._detect_deadline - time.time()
            sys.stdout.write(
                f"\r[S2] waiting for re-detection ({remaining:.1f}s left)   "
            )
            sys.stdout.flush()
            if time.time() > self._detect_deadline:
                rprint("\n[S2] object lost — restarting search")
                self.ctrl.locked_joints = set()
                self._yaw_iters = 0
                self.detector.clear()
                self.state = State.SEARCH
        return None

    # ── Stage 3: Height ──────────────────────────────────────────────────────

    def _height_detect(self) -> None:
        result = self.detector.latest
        if result is not None:
            self._p_obj = result[0]

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
                self._detect_deadline = time.time() + self.cfg.detect_timeout_s
                self.state   = State.HEIGHT_DETECT

            self._begin_move(end_pos, tgt_rot, duration, _after)
        else:
            remaining = self._detect_deadline - time.time()
            sys.stdout.write(
                f"\r[S3] no detection ({remaining:.1f}s left)   "
            )
            sys.stdout.flush()
            if time.time() > self._detect_deadline:
                rprint("\n[S3] object lost — restarting search")
                self.ctrl.locked_joints = set()
                self._yaw_iters = 0
                self.detector.clear()
                self.state = State.SEARCH
        return None

    # ── Stage 4: Approach ────────────────────────────────────────────────────

    def _launch_approach(self):
        fb = read_gripper_feedback(self.ctrl.piper)
        self._gripper_open_cmd  = fb["angle_raw"]   # start from current position
        self._gripper_ready_t   = 0.0               # start stepping immediately
        self.ctrl._gripper_open = True
        self.state              = State.OPEN_GRIPPER

    def _open_gripper_wait(self) -> None:
        now = time.time()
        if now < self._gripper_ready_t:
            return None

        # Step the gripper open incrementally until fully open.
        if self._gripper_open_cmd < self.ctrl.GRIPPER_OPEN_ANGLE:
            self._gripper_open_cmd = min(
                self.ctrl.GRIPPER_OPEN_ANGLE,
                self._gripper_open_cmd + self.cfg.gripper_open_step_raw,
            )
            self.ctrl.piper.GripperCtrl(
                self._gripper_open_cmd, self.ctrl.GRIPPER_EFFORT, 0x01, 0)
            self._gripper_ready_t = now + self._GRIP_SETTLE
            return None
        # Rotation is already what Stage 3 converged to — keep it exactly.
        tgt_rot    = self.ctrl.target_rot.copy()
        tcp_offset = self.ctrl.get_tcp_offset_base()
        tcp_mag    = np.linalg.norm(tcp_offset)

        # Approach direction: XY unit vector pointing from base toward object.
        obj_xy     = np.array([self._p_obj[0], self._p_obj[1], 0.0])
        obj_xy_mag = np.linalg.norm(obj_xy)
        approach_dir = obj_xy / obj_xy_mag if obj_xy_mag > 1e-6 else np.array([1.0, 0.0, 0.0])

        # Pull back by pregrasp_offset so the TCP stops just before the object.
        grasp_tcp  = np.array([self._p_obj[0], self._p_obj[1],
                                self.ctrl.target_pos[2]]) \
                     - approach_dir * self.cfg.pregrasp_offset_m
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
                    f"[GRIP] contact!  "
                    f"angle={fb['angle_mm']:.1f} mm  "
                    f"effort={fb['effort_nm']:+.2f} N·m  — squeezing"
                )
                squeeze_cmd = max(
                    self.cfg.gripper_min_hold_angle_raw,
                    self._grip_cmd - self.cfg.gripper_squeeze_extra_raw,
                )
                self.ctrl.piper.GripperCtrl(
                    squeeze_cmd, self.ctrl.GRIPPER_EFFORT, 0x01, 0)
                time.sleep(self._GRIP_SETTLE * 3)   # hold squeeze briefly
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
        rprint("[LIFT] complete — moving to intermediate")
        self.ctrl.locked_joints = set()
        self.state = State.PLACE

        def _after_place():
            rprint("[PLACE] at place pose — releasing object")
            self.state = State.RELEASE
            self.ctrl.piper.GripperCtrl(
                self.ctrl.GRIPPER_OPEN_ANGLE, self.ctrl.GRIPPER_EFFORT, 0x01, 0)
            self.ctrl._gripper_open = True
            time.sleep(0.5)
            self.success = True
            self.state   = State.DONE

        def _after_intermediate():
            rprint("[PLACE] at intermediate — moving to place pose")
            self.state = State.PLACE
            place_deg = [v / 1000.0 for v in self.cfg.place_pose]
            self._begin_move_joints(place_deg, 3.0, _after_place)

        intermediate_deg = [v / 1000.0 for v in self.cfg.intermediate_pose]
        self._begin_move_joints(intermediate_deg, 2.0, _after_intermediate)


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

            # Reload config from disk so edits to config.py take effect immediately.
            import importlib
            import config as _config_module
            importlib.reload(_config_module)
            cfg = _config_module.GraspConfig()
            machine.cfg = cfg
            detector.cfg = cfg
            detector._queue     = collections.deque(maxlen=cfg.detection_window)
            detector._min_count = cfg.detection_frames
            print("Config reloaded.")

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
