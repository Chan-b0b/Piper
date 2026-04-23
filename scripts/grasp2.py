#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Autonomous grasping pipeline v2 — GroundingDINO + ZED-X + Piper arm.

Pipeline
--------
  Stage 1 — Detect  : wait at search pose for N consecutive detections
  Stage 2 — Yaw     : rotate joint1 toward object; re-detect; repeat until
                       yaw error < threshold (lost object → back to Stage 1)
  Stage 3 — Height  : fresh detection → move EE to object Z while keeping
                       RX=0 RY=90° RZ=yaw (gripper horizontal, facing object)
  Stage 4 — Grasp   : horizontal approach → close gripper → lift → home

Usage
-----
  python3 scripts/grasp2.py [--classes "can"] [--can can2] [--conf 0.25] [--vis]

Keys (during pipeline)
  q / Ctrl+C : abort at any time and return home
"""

import argparse
import os
import select
import sys
import termios
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
    close_gripper_until_grasp,
)
from ik_ee_ctrl import IKEndEffectorCtrl, CONTROL_DT


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
    """Non-blocking check for q/Ctrl-C. Caller must already be in raw-terminal mode.
    Raises KeyboardInterrupt immediately on Ctrl-C so the program exits at once."""
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = os.read(sys.stdin.fileno(), 4).decode("utf-8", errors="replace")
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch in ("q", "Q")
    return False


def rprint(msg: str):
    """Print in raw-terminal mode (appends \\r before \\n)."""
    sys.stdout.write(msg.rstrip("\n") + "\r\n")
    sys.stdout.flush()


# ── Geometry ──────────────────────────────────────────────────────────────────

def approach_rotation(yaw_rad: float) -> np.ndarray:
    """
    Desired EE orientation for a horizontal gripper approach.

    yaw_rad should come from ctrl.q[0] (joint1 angle) — Stage 2 already
    rotated joint1 to face the object, so this just tilts the gripper
    to be horizontal (RY=90°) while keeping that yaw (RZ=joint1).

    RPY = (RX=0, RY=90°, RZ=yaw_rad)  →  R = Rz(yaw) · Ry(π/2)
    """
    return pin.rpy.rpyToMatrix(0.0, np.pi / 2, yaw_rad)


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_once(
    cam, rgb_mat, depth_mat, runtime, K,
    processor, det_model, device, text_prompt,
    piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
    cfg: GraspConfig,
    vis_win: str | None = None,
) -> tuple[np.ndarray, float] | None:
    """
    Grab one ZED frame, run GroundingDINO, back-project the best detection.

    Returns (p_base, conf) for the highest-confidence box above cfg.min_conf,
    or None if nothing found.
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
        cy = y1 + (2.0 / 3.0) * (y2 - y1)   # sample 1/3 up from bottom of bbox
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
        cv2.imshow(vis_win, frame)
        cv2.waitKey(1)

    return (best_p, best_conf) if best_p is not None else None


def collect_detections(
    n: int,
    detect_fn,
    miss_limit: int = 30,
) -> np.ndarray | None:
    """
    Accumulate n consecutive detections (any miss resets the counter).
    Returns median base-frame position, or None after miss_limit misses.
    """
    hits:  list = []
    misses: int = 0
    while len(hits) < n:
        result = detect_fn()
        if result is not None:
            hits.append(result[0])
            misses = 0
        else:
            misses += 1
            if misses >= miss_limit:
                return None
    return np.median(np.array(hits), axis=0)


# ── Arm motion helpers ────────────────────────────────────────────────────────

def smooth_to_joints(ctrl: IKEndEffectorCtrl, sdk_values: list, label: str = "moving"):
    """
    IK-based smooth interpolation from the current pose to the given SDK joint
    configuration, then sync ctrl internal state to match.
    """
    q_target = pin.neutral(ctrl.model)
    for i in range(6):
        q_target[i] = np.deg2rad(sdk_values[i] / 1000.0)
    q_target[6] = q_target[7] = 0.0
    pin.forwardKinematics(ctrl.model, ctrl.data, q_target)
    pin.updateFramePlacements(ctrl.model, ctrl.data)
    T = ctrl.data.oMf[ctrl.ee_id]
    dp = T.translation - ctrl.target_pos
    dR = ctrl.target_rot.T @ T.rotation
    ctrl._smooth_move(dp, dR, label)
    ctrl.q          = q_target.copy()
    ctrl.target_pos = T.translation.copy()
    ctrl.target_rot = T.rotation.copy()


def return_to_home(ctrl: IKEndEffectorCtrl, cfg: GraspConfig):
    """Smoothly: current → joint1=0 → intermediate → zero; open gripper."""
    print("Zeroing joint1…")
    ctrl.locked_joints = set()   # ensure joint1 is free to move
    j1_zero = [round(np.rad2deg(ctrl.q[i]) * 1000) for i in range(6)]
    j1_zero[0] = 0
    smooth_to_joints(ctrl, j1_zero, "zero joint1")
    print("Returning to intermediate…")
    smooth_to_joints(ctrl, cfg.intermediate_pose, "intermediate")
    print("Returning to zero…")
    q_zero = pin.neutral(ctrl.model)
    q_zero[6] = q_zero[7] = 0.0
    pin.forwardKinematics(ctrl.model, ctrl.data, q_zero)
    pin.updateFramePlacements(ctrl.model, ctrl.data)
    T = ctrl.data.oMf[ctrl.ee_id]
    dp = T.translation - ctrl.target_pos
    dR = ctrl.target_rot.T @ T.rotation
    ctrl._smooth_move(dp, dR, "return zero")
    ctrl.q          = q_zero.copy()
    ctrl.target_pos = T.translation.copy()
    ctrl.target_rot = T.rotation.copy()
    # ctrl.piper.GripperCtrl(ctrl.GRIPPER_OPEN_ANGLE, ctrl.GRIPPER_EFFORT, 0x01, 0)
    # ctrl._gripper_open = True


# ── Stage 1 — Detect at search pose ──────────────────────────────────────────

def stage1_detect(
    ctrl: IKEndEffectorCtrl,
    detect_fn,
    cfg: GraspConfig,
) -> np.ndarray | None:
    """
    Spin at the search pose until cfg.detection_frames consecutive detections
    are accumulated.

    Returns median base-frame object position, or None if q pressed.
    """
    rprint("[Stage 1] Waiting for detection…  q = abort")
    hits:  list = []
    misses: int = 0

    while True:
        if quit_pressed():
            return None

        result = detect_fn()
        if result is not None:
            p, conf = result
            hits.append(p)
            misses = 0
            sys.stdout.write(
                f"\r  Detection {len(hits)}/{cfg.detection_frames}  "
                f"conf={conf:.2f}  "
                f"X={p[0]*100:+.1f} Y={p[1]*100:+.1f} Z={p[2]*100:+.1f} cm   "
            )
            sys.stdout.flush()
            if len(hits) >= cfg.detection_frames:
                median_p = np.median(np.array(hits), axis=0)
                rprint(
                    f"\n  Locked: "
                    f"X={median_p[0]*100:+.2f} Y={median_p[1]*100:+.2f} "
                    f"Z={median_p[2]*100:+.2f} cm"
                )
                return median_p
        else:
            hits.clear()
            misses += 1
            sys.stdout.write(f"\r  No detection (miss {misses})   ")
            sys.stdout.flush()


# ── Stage 2 — Yaw alignment ───────────────────────────────────────────────────

def stage2_yaw(
    ctrl: IKEndEffectorCtrl,
    p_obj: np.ndarray,
    detect_fn,
    cfg: GraspConfig,
) -> np.ndarray | None:
    """
    Rotate joint1 toward the object, re-detect, repeat until yaw error is
    within cfg.yaw_threshold_deg.

    Returns refined p_obj, or None if the object is lost (caller retries Stage 1).
    """
    rprint("\n[Stage 2] Aligning yaw…")

    for iteration in range(cfg.yaw_max_iters):
        if quit_pressed():
            return None

        ideal_yaw_deg   = np.rad2deg(np.arctan2(p_obj[1], p_obj[0]))
        current_yaw_deg = np.rad2deg(ctrl.q[0])
        yaw_err         = abs(ideal_yaw_deg - current_yaw_deg)

        rprint(
            f"  Iter {iteration + 1}: "
            f"ideal={ideal_yaw_deg:.1f}°  current={current_yaw_deg:.1f}°  "
            f"err={yaw_err:.1f}°"
        )

        if yaw_err <= cfg.yaw_threshold_deg:
            rprint("  Yaw aligned.")
            return p_obj

        # Clamp to joint1 physical limits (±154°)
        target_yaw_deg = float(np.clip(ideal_yaw_deg, -154.0, 154.0))
        yawed_pose     = list(cfg.search_pose)
        yawed_pose[0]  = int(target_yaw_deg * 1000)
        smooth_to_joints(ctrl, yawed_pose, "yaw align")

        rprint("  Re-detecting…")
        new_p = collect_detections(cfg.detection_frames, detect_fn)
        if new_p is None:
            rprint("  Object lost during yaw alignment.")
            return None
        p_obj = new_p

    rprint(f"  Yaw did not converge in {cfg.yaw_max_iters} iterations — proceeding.")
    return p_obj


# ── Stage 3 — Height alignment ────────────────────────────────────────────────

def stage3_height(
    ctrl: IKEndEffectorCtrl,
    p_obj: np.ndarray,
    detect_fn,
    cfg: GraspConfig,
) -> np.ndarray | None:
    """
    1. Fresh detection to update the height target.
    2. Move the EE to the object's Z while maintaining gripper orientation:
         RX=0, RY=90°, RZ=joint1  (horizontal approach orientation).
    Each iteration: re-detect → compute clamped Z + rotation delta →
    _smooth_move (smooth interpolation) → check convergence.

    Returns the latest p_obj when both height and orientation have converged,
    or None on timeout / q pressed.
    """
    rprint("\n[Stage 3] Aligning height…")

    # Fresh detection to get accurate target position
    rprint("  Running fresh detection…")
    new_p = collect_detections(cfg.detection_frames, detect_fn)
    if new_p is not None:
        p_obj = new_p
    rprint(
        f"  Height target: Z={p_obj[2]*100:+.2f} cm  "
        f"(X={p_obj[0]*100:+.2f} Y={p_obj[1]*100:+.2f})"
    )

    # Joint1 is locked — yaw is fixed for the rest of stage 3.
    tgt_rot  = approach_rotation(ctrl.q[0])
    deadline = time.time() + cfg.align_timeout_s

    orig_duration      = ctrl.move_duration
    ctrl.move_duration = cfg.align_timeout_s / 10   # each segment ≈ 1/10 of timeout

    try:
        while time.time() < deadline:
            if quit_pressed():
                return None

            z_err       = p_obj[2] - ctrl.target_pos[2]
            dR_rem      = ctrl.target_rot.T @ tgt_rot
            omega       = pin.log3(dR_rem)
            rot_err_deg = np.rad2deg(np.linalg.norm(omega))

            rprint(
                f"  dZ={z_err*100:+.1f}cm  rot_err={rot_err_deg:.1f}°"
            )

            if abs(z_err) < cfg.height_threshold_m and rot_err_deg < cfg.orient_threshold_deg:
                rprint("  Height aligned.")
                return p_obj

            # Clamp the step and move smoothly
            dz = float(np.clip(z_err, -cfg.align_z_step_m * 10, cfg.align_z_step_m * 10))
            dp = np.array([0.0, 0.0, dz])
            dR = dR_rem   # rotate fully toward target each segment
            ctrl._smooth_move(dp, dR, "align height")

            # Re-detect after moving to update Z target
            result = detect_fn()
            if result is not None:
                p_obj = result[0]

    finally:
        ctrl.move_duration = orig_duration

    rprint("\n  Height alignment timeout.")
    return None


# ── Stage 4 — Approach and grasp ──────────────────────────────────────────────

def stage4_grasp(
    ctrl: IKEndEffectorCtrl,
    p_obj: np.ndarray,
    cfg: GraspConfig,
) -> bool:
    """
    Pure horizontal approach → close gripper with contact detection → lift.

    The EE Z stays fixed at the height already aligned in Stage 3.
    Returns True on successful grasp, False on abort or missed grasp.
    """
    rprint("\n[Stage 4] Approach and grasp…")

    # Stage 3 already converged ctrl.target_rot to the approach orientation —
    # reuse it directly so the gripper keeps exactly the same RPY.
    tgt_rot = ctrl.target_rot.copy()

    # TCP offset: link6 origin → gripper midpoint along approach axis
    tcp_mag        = np.linalg.norm(ctrl.get_tcp_offset_base())
    tcp_offset_vec = tgt_rot[:, 2] * tcp_mag   # along gripper Z-axis (approach)

    # Keep current Z (already aligned); XY from object position
    grasp_tcp = np.array([p_obj[0], p_obj[1], ctrl.target_pos[2]])
    grasp_pos = grasp_tcp - tcp_offset_vec

    orig_duration      = ctrl.move_duration
    ctrl.move_duration = cfg.approach_duration_s

    try:
        # 1. Open gripper
        rprint("[1/4] Opening gripper…")
        ctrl.piper.GripperCtrl(ctrl.GRIPPER_OPEN_ANGLE, ctrl.GRIPPER_EFFORT, 0x01, 0)
        ctrl._gripper_open = True
        time.sleep(0.5)
        if quit_pressed():
            return False

        # 2. Advance slowly to grasp position
        rprint("[2/4] Advancing to grasp position…")
        dp = grasp_pos - ctrl.target_pos
        ctrl._smooth_move(dp, np.eye(3), "advance")
        if quit_pressed():
            return False

        # 3. Close gripper with contact detection
        rprint("[3/4] Closing gripper…")
        grabbed, grip_fb = close_gripper_until_grasp(
            ctrl.piper,
            start_angle_raw=ctrl.GRIPPER_OPEN_ANGLE,
            effort=ctrl.GRIPPER_EFFORT,
            min_angle_raw=cfg.gripper_min_hold_angle_raw,
            step_angle_raw=cfg.gripper_close_step_raw,
            contact_effort_raw=cfg.gripper_contact_effort_raw,
            timeout=cfg.gripper_close_timeout_s,
        )
        time.sleep(0.5)
        ctrl._gripper_open = False

        if not grabbed:
            rprint(
                f"\nNo object in gripper: "
                f"angle={grip_fb['angle_mm']:.1f}mm  "
                f"effort={grip_fb['effort_nm']:+.2f}N·m"
            )
            return False

        rprint(
            f"\nGrasped!  "
            f"angle={grip_fb['angle_mm']:.1f}mm  "
            f"effort={grip_fb['effort_nm']:+.2f}N·m"
        )

        # 4. Lift
        rprint("[4/4] Lifting…")
        ctrl._smooth_move(np.array([0.0, 0.0, cfg.lift_height_m]), np.eye(3), "lift")
        rprint("Grasp complete.")
        return True

    finally:
        ctrl.move_duration = orig_duration


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Piper autonomous grasp pipeline v2")
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

    # ── Calibration ───────────────────────────────────────────────────────────
    if not os.path.exists(cfg.calib_path):
        print(f"ERROR: calibration file not found: {cfg.calib_path}")
        sys.exit(1)
    T_ee_cam = np.load(cfg.calib_path)

    # ── FK model ──────────────────────────────────────────────────────────────
    fk_model, fk_data, fk_ee_id = build_fk()

    # ── Piper arm ─────────────────────────────────────────────────────────────
    piper = C_PiperInterface_V2(cfg.can_port)
    piper.ConnectPort()
    time.sleep(0.2)

    ctrl = IKEndEffectorCtrl(piper, move_duration=2.0)
    ctrl.emergency_restore()
    ctrl.enable()
    ctrl.go_to_joints([0, 0, 0, 0, 0, 0])   # go to zero and sync pinocchio state

    # ── ZED camera ────────────────────────────────────────────────────────────
    print("Opening ZED camera…")
    cam, rgb_mat, depth_mat, runtime, K, _ = open_zed(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        depth_min_m=cfg.depth_min_m,
        depth_max_m=cfg.depth_max_m,
    )
    print("ZED opened.")

    # ── GroundingDINO ─────────────────────────────────────────────────────────
    print(f"Loading {cfg.detection_model}…")
    processor = AutoProcessor.from_pretrained(cfg.detection_model)
    det_model = AutoModelForZeroShotObjectDetection.from_pretrained(cfg.detection_model)
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    det_model.to(device)
    print(f"Detection model ready on {device}.")

    vis_win = "Grasp" if args.vis else None

    def detect():
        return detect_once(
            cam, rgb_mat, depth_mat, runtime, K,
            processor, det_model, device, text_prompt,
            piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
            cfg, vis_win,
        )

    # ── Main pick loop ────────────────────────────────────────────────────────
    try:
        while True:
            if not wait_for_space():
                break

            print("\nMoving to search pose…")
            smooth_to_joints(ctrl, cfg.search_pose, "search pose")
            print("At search pose.\n")

            orig = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            success = False
            try:
                while True:   # inner retry loop: Stage 1 → 2 → 3 → 4
                    # Stage 1: Detect at search pose
                    p_obj = stage1_detect(ctrl, detect, cfg)
                    if p_obj is None:
                        rprint("\nAborted.")
                        break

                    # Safety: distance check
                    dist_m = float(np.linalg.norm(p_obj))
                    if dist_m > cfg.max_reach_m:
                        rprint(
                            f"\nObject too far ({dist_m*100:.1f}cm > "
                            f"{cfg.max_reach_m*100:.0f}cm) — retrying detection."
                        )
                        continue

                    # Stage 2: Yaw alignment (lost object → retry Stage 1)
                    p_obj = stage2_yaw(ctrl, p_obj, detect, cfg)
                    if p_obj is None:
                        rprint("\nObject lost — returning to search pose.")
                        smooth_to_joints(ctrl, cfg.search_pose, "search pose")
                        continue

                    # Stage 3 & 4: lock joint1 so IK won't disturb the yaw
                    ctrl.locked_joints = {0}
                    try:
                        # Stage 3: Height alignment (failure → retry Stage 1)
                        p_obj = stage3_height(ctrl, p_obj, detect, cfg)
                        if p_obj is None:
                            rprint("\nHeight alignment failed — returning to search pose.")
                            smooth_to_joints(ctrl, cfg.search_pose, "search pose")
                            continue

                        # Stage 4: Approach and grasp
                        success = stage4_grasp(ctrl, p_obj, cfg)
                    finally:
                        ctrl.locked_joints = set()
                    break   # exit retry loop regardless of grasp outcome

            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig)

            if success:
                print("\nGrasp succeeded!")
            else:
                print("\nGrasp failed or aborted.")
            return_to_home(ctrl, cfg)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
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
