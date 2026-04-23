#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Keyboard IK jog + live GroundingDINO detection with delta guidance.

Move the arm with the keyboard (same bindings as ik_ee_ctrl.py) while the
ZED + GroundingDINO pipeline continuously detects the target object and prints:

  [EE]  current end-effector position in the base frame
  [OBJ] detected object position in the base frame
  [Δ]   delta = object − EE  (how far and which direction to jog)

Use the delta info to manually jog the gripper toward the object.
If the gripper aligns with the object as the delta approaches zero you
know the hand-eye calibration is correct.

Usage
-----
  python3 scripts/jog_and_detect.py --classes "plastic cup" --can can2 [--vis]

Key bindings  (same as ik_ee_ctrl.py)
  ↑  / ↓    : Z+ / Z−   (mm)
  ← / →     : X− / X+
  w  / s    : Y+ / Y−
  u  / j    : RX+ / RX−  (body frame)
  i  / k    : RY+ / RY−
  o  / l    : RZ+ / RZ−
  c         : Toggle gripper
  + / -     : Step size up / down
  q / Ctrl+C: Quit
"""

import os
import sys
import tty
import termios
import select
import time
import argparse
import threading

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

import numpy as np
import cv2
import pinocchio as pin
import pyzed.sl as sl
import torch
from piper_sdk import C_PiperInterface_V2
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from utils import (
    build_fk, open_zed, sample_depth, backproject,
    read_gripper_feedback, close_gripper_until_grasp,
    URDF_PATH, EE_FRAME,
)
from ik_ee_ctrl import IKEndEffectorCtrl, SPEED, FACTOR, JOINT_LIMITS, CONTROL_DT

# ── Constants ─────────────────────────────────────────────────────────────────
DEPTH_PATCH = 5
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0

# Gripper feedback parameters (same as grasp.py)
GRIPPER_CLOSE_STEP_RAW    = 4000
GRIPPER_CONTACT_EFFORT_RAW = 120
GRIPPER_MIN_HOLD_ANGLE_RAW = 3000
GRIPPER_CLOSE_TIMEOUT_S    = 10.0


# ── Shared detection result (thread-safe) ─────────────────────────────────────
class DetectionState:
    def __init__(self):
        self._lock   = threading.Lock()
        self.p_base  = None    # np.ndarray (3,) metres, or None
        self.label   = ""
        self.conf    = 0.0
        self.depth_m = 0.0

    def update(self, p_base, label, conf, depth_m):
        with self._lock:
            self.p_base  = p_base.copy()
            self.label   = label
            self.conf    = conf
            self.depth_m = depth_m

    def get(self):
        with self._lock:
            return (
                self.p_base.copy() if self.p_base is not None else None,
                self.label, self.conf, self.depth_m,
            )


# ── Detection thread ──────────────────────────────────────────────────────────
def detection_thread_fn(
    cam, rgb_mat, depth_mat, runtime, K,
    processor, model, text_prompt, device,
    ctrl,        # IKEndEffectorCtrl — read target_pos/target_rot (no piper calls)
    T_ee_cam,
    state: DetectionState,
    stop_event: threading.Event,
    vis: bool,
    conf_thresh: float,
):
    """
    Continuously grab ZED frames, run GroundingDINO, and write the best
    detection's base-frame position into `state`.

    Uses ctrl.target_pos / ctrl.target_rot (IK target) to build T_base_ee
    without reading from the piper SDK, avoiding cross-thread SDK calls.
    """
    while not stop_event.is_set():
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.01)
            continue

        cam.retrieve_image(rgb_mat,    sl.VIEW.LEFT)
        cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        frame = np.ascontiguousarray(rgb_mat.get_data()[:, :, :3])
        depth = depth_mat.get_data()

        # GroundingDINO inference
        inputs = processor(images=frame, text=text_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[(frame.shape[0], frame.shape[1])]
        )[0]
        
        # Filter by confidence threshold
        mask = results["scores"] >= conf_thresh
        results = {
            "boxes": results["boxes"][mask],
            "scores": results["scores"][mask],
            "labels": [results["labels"][i] for i in range(len(mask)) if mask[i]]
        }

        # Snapshot EE pose from IK controller (benign race: one frame stale at worst)
        ee_pos = ctrl.target_pos
        ee_rot = ctrl.target_rot
        if ee_pos is None or ee_rot is None:
            continue
        T_be = np.eye(4)
        T_be[:3, :3] = ee_rot.copy()
        T_be[:3,  3] = ee_pos.copy()

        best_p    = None
        best_conf = 0.0
        best_lbl  = ""
        best_d    = 0.0

        for box, score, label_text in zip(results["boxes"], results["scores"], results["labels"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            conf_val = float(score)

            cx = (x1 + x2) / 2.0  # horizontal center
            cy = y1 + (2.0/3.0) * (y2 - y1)  # 1/3 from bottom (more likely to hit object vs air)

            depth_m = sample_depth(depth, cx, cy, patch=DEPTH_PATCH,
                                   depth_min_m=DEPTH_MIN_M, depth_max_m=DEPTH_MAX_M)
            if depth_m is None:
                continue
            
            # Add 6cm to depth so gripper reaches further into the object
            depth_m += 0.06

            p_cam  = backproject(cx, cy, depth_m, K)
            p_base = (T_be @ T_ee_cam @ np.append(p_cam, 1.0))[:3]

            if conf_val > best_conf:
                best_conf = conf_val
                best_p    = p_base
                best_lbl  = label_text
                best_d    = depth_m

            if vis:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_text} {conf_val:.2f} | d={depth_m*100:.1f}cm",
                            (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)

        if best_p is not None:
            state.update(best_p, best_lbl, best_conf, best_d)

        if vis:
            cv2.imshow("ZED + GroundingDINO", frame)
            cv2.waitKey(1)


# ── 4-line status display ─────────────────────────────────────────────────────
_status_initialised = False


def print_status(ctrl: IKEndEffectorCtrl, state: DetectionState, label: str):
    """
    Overwrite a fixed 4-line block in the terminal:
      Line 1  [EE/wrist]  link6 pose (IK control frame)
      Line 2  [TCP/tip]   link7 fingertip position
      Line 3  [OBJ]       detected object position
      Line 4  [Δ tip→obj] delta fingertip → object (jog this to zero)
    """
    global _status_initialised

    ee_mm  = ctrl.target_pos * 1000.0
    rpy    = np.rad2deg(pin.rpy.matrixToRpy(ctrl.target_rot))
    grip   = "OPEN  " if ctrl._gripper_open else "CLOSED"
    step_s = f"{ctrl.step_mm:.0f}mm/{ctrl.step_deg:.0f}°"

    line1 = (
        f"\033[2K\r[EE/wrist] "
        f"X={ee_mm[0]:+7.1f}  Y={ee_mm[1]:+7.1f}  Z={ee_mm[2]:+7.1f} mm  "
        f"RX={rpy[0]:+5.1f}° RY={rpy[1]:+5.1f}° RZ={rpy[2]:+5.1f}°  "
        f"grip={grip}  step={step_s}  [{label}]"
    )

    tcp_mm = ctrl.get_tcp_pos() * 1000.0
    line2 = (
        f"\033[2K\r[TCP/tip]  "
        f"X={tcp_mm[0]:+7.1f}  Y={tcp_mm[1]:+7.1f}  Z={tcp_mm[2]:+7.1f} mm"
    )

    p_obj, obj_lbl, obj_conf, obj_depth = state.get()

    if p_obj is not None:
        obj_mm   = p_obj * 1000.0
        delta_mm = obj_mm - tcp_mm   # delta from fingertip to object
        dist     = np.linalg.norm(delta_mm)

        hints = []
        if delta_mm[0] >  3: hints.append("→ X+")
        elif delta_mm[0] < -3: hints.append("← X−")
        if delta_mm[1] >  3: hints.append("w Y+")
        elif delta_mm[1] < -3: hints.append("s Y−")
        if delta_mm[2] >  3: hints.append("↑ Z+")
        elif delta_mm[2] < -3: hints.append("↓ Z−")
        hint_str = "  keys: " + " ".join(hints) if hints else "  *** ALIGNED ***"

        line3 = (
            f"\033[2K\r[OBJ]      "
            f"X={obj_mm[0]:+7.1f}  Y={obj_mm[1]:+7.1f}  Z={obj_mm[2]:+7.1f} mm  "
            f"conf={obj_conf:.2f}  depth={obj_depth*100:.1f}cm  [{obj_lbl}]"
        )
        line4 = (
            f"\033[2K\r[Δ tip→obj]"
            f"ΔX={delta_mm[0]:+7.1f}  ΔY={delta_mm[1]:+7.1f}  ΔZ={delta_mm[2]:+7.1f} mm  "
            f"dist={dist:.1f}mm{hint_str}"
        )
    else:
        line3 = "\033[2K\r[OBJ]      -- no detection --"
        line4 = "\033[2K\r[Δ tip→obj]--"

    if not _status_initialised:
        sys.stdout.write("\r\n\r\n\r\n\r\n\033[4A")
        _status_initialised = True

    # Print 4 lines then move cursor back to top of the block
    sys.stdout.write(f"{line1}\r\n{line2}\r\n{line3}\r\n{line4}\033[3A\r")
    sys.stdout.flush()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Jog arm with keyboard + live detection delta")
    parser.add_argument("--classes", default="Can",
                        help="Comma-separated object class names")
    parser.add_argument("--can",    default="can2")
    parser.add_argument("--calib",  default="/workspace/calibration/T_ee_cam.npy")
    parser.add_argument("--model",  default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--vis",    action="store_true", help="Show OpenCV detection window")
    parser.add_argument("--step-mm",  type=float, default=20.0)
    parser.add_argument("--step-deg", type=float, default=5.0)
    args = parser.parse_args()

    class_list = [c.strip() for c in args.classes.split(",")]
    text_prompt = ". ".join(class_list) + "."
    print(f"Target classes: {class_list}")
    print(f"Text prompt: {text_prompt}")

    if not os.path.exists(args.calib):
        print(f"ERROR: calibration file not found: {args.calib}")
        sys.exit(1)
    T_ee_cam = np.load(args.calib)
    print(f"T_ee_cam loaded.")

    # ── Arm ───────────────────────────────────────────────────────────────────
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    time.sleep(0.2)

    ctrl = IKEndEffectorCtrl(piper, step_mm=args.step_mm, step_deg=args.step_deg)
    ctrl.emergency_restore()
    ctrl.enable()
    ctrl.go_to_zero_and_sync()

    # ── ZED ───────────────────────────────────────────────────────────────────
    print("Opening ZED camera…")
    cam, rgb_mat, depth_mat, runtime, K, _ = open_zed(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        depth_min_m=DEPTH_MIN_M,
        depth_max_m=DEPTH_MAX_M,
    )
    print("ZED opened.")

    # ── GroundingDINO─────────────────────────────────────────────────────────────────
    print(f"Loading GroundingDINO: {args.model}")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model ready on {device}.")

    # ── Detection thread ──────────────────────────────────────────────────────────
    det_state  = DetectionState()
    stop_event = threading.Event()
    det_thread = threading.Thread(
        target=detection_thread_fn,
        args=(
            cam, rgb_mat, depth_mat, runtime, K,
            processor, model, text_prompt, device,
            ctrl, T_ee_cam,
            det_state, stop_event,
            args.vis, args.conf,
        ),
        daemon=True,
    )
    det_thread.start()

    # ── Keyboard loop ─────────────────────────────────────────────────────────
    print("\n=== Jog & Detect ===")
    print("  Arrow keys / w s : translate EE")
    print("  u j / i k / o l  : rotate EE (body frame)")
    print("  c                 : toggle gripper")
    print("  + / -             : step size")
    print("  q / Ctrl+C        : quit\n")

    ctrl._set_raw()
    try:
        while True:
            key = ctrl._read_key()

            dm      = ctrl.step_mm / 1000.0
            dr      = np.deg2rad(ctrl.step_deg)
            label   = ""
            dp      = np.zeros(3)
            dR      = np.eye(3)
            do_move = True

            if   key == '\x1b[A':      dp[2] =  dm; label = "↑ Z+"
            elif key == '\x1b[B':      dp[2] = -dm; label = "↓ Z−"
            elif key == '\x1b[C':      dp[0] =  dm; label = "→ X+"
            elif key == '\x1b[D':      dp[0] = -dm; label = "← X−"
            elif key in ('w', 'W'):    dp[1] =  dm; label = "w Y+"
            elif key in ('s', 'S'):    dp[1] = -dm; label = "s Y−"

            elif key in ('u', 'U'):    dR = pin.exp3(np.array([ dr, 0, 0])); label = "u RX+"
            elif key in ('j', 'J'):    dR = pin.exp3(np.array([-dr, 0, 0])); label = "j RX−"
            elif key in ('i', 'I'):    dR = pin.exp3(np.array([0,  dr, 0])); label = "i RY+"
            elif key in ('k', 'K'):    dR = pin.exp3(np.array([0, -dr, 0])); label = "k RY−"
            elif key in ('o', 'O'):    dR = pin.exp3(np.array([0, 0,  dr])); label = "o RZ+"
            elif key in ('l', 'L'):    dR = pin.exp3(np.array([0, 0, -dr])); label = "l RZ−"

            elif key == '+':
                ctrl.step_mm  = min(ctrl.step_mm  + 2.0, 50.0)
                ctrl.step_deg = min(ctrl.step_deg + 1.0, 20.0)
                label = "step ↑"; do_move = False
            elif key == '-':
                ctrl.step_mm  = max(ctrl.step_mm  - 2.0, 1.0)
                ctrl.step_deg = max(ctrl.step_deg - 1.0, 1.0)
                label = "step ↓"; do_move = False

            elif key in ('c', 'C'):
                if ctrl._gripper_open:
                    # Close with feedback
                    label = "closing…"
                    print_status(ctrl, det_state, label)
                    grabbed, grip_fb = close_gripper_until_grasp(
                        ctrl.piper,
                        start_angle_raw=ctrl.GRIPPER_OPEN_ANGLE,
                        effort=ctrl.GRIPPER_EFFORT,
                        min_angle_raw=GRIPPER_MIN_HOLD_ANGLE_RAW,
                        step_angle_raw=GRIPPER_CLOSE_STEP_RAW,
                        contact_effort_raw=GRIPPER_CONTACT_EFFORT_RAW,
                        timeout=GRIPPER_CLOSE_TIMEOUT_S,
                    )
                    ctrl._gripper_open = False
                    if grabbed:
                        label = f"GRABBED: {grip_fb['angle_mm']:.1f}mm {grip_fb['effort_nm']:+.2f}Nm"
                    else:
                        label = f"CLOSED: {grip_fb['angle_mm']:.1f}mm {grip_fb['effort_nm']:+.2f}Nm"
                else:
                    # Open
                    ctrl.piper.GripperCtrl(ctrl.GRIPPER_OPEN_ANGLE, ctrl.GRIPPER_EFFORT, 0x01, 0)
                    ctrl._gripper_open = True
                    label = "grip OPEN"
                print_status(ctrl, det_state, label)
                continue

            elif key in ('q', 'Q', '\x03'):
                break

            elif not key:
                # No keypress — refresh display at ~10 Hz
                print_status(ctrl, det_state, "")
                time.sleep(0.1)
                continue

            else:
                continue

            if do_move:
                ctrl._smooth_move(dp, dR, label)
            print_status(ctrl, det_state, label)

    finally:
        stop_event.set()
        ctrl._restore_terminal()
        # Move past the 4-line status block
        sys.stdout.write("\r\n\r\n\r\n\r\n")
        sys.stdout.flush()
        cam.close()
        if args.vis:
            cv2.destroyAllWindows()
        print("Returning to zero…")
        INTERMEDIATE = [0, 34196, -32149, 0, 32955, 0]
        ctrl.go_to_joints(INTERMEDIATE)
        ctrl.go_to_zero_and_sync()
        ctrl.emergency_stop()


if __name__ == "__main__":
    main()
