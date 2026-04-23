#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Autonomous grasping pipeline — GroundingDINO + ZED-X + Piper arm.

Pipeline
--------
  Stage 1 — Sweep    : scan with XYZ offsets until N consecutive detections
  Stage 2 — Yaw      : rotate joint1 to face object; repeat until within threshold
  Stage 3 — Align EE : bring link6 to object height; rotate gripper to horizontal
                        level jaws; detection keeps updating the target Z live
  Stage 4 — Grasp    : pure horizontal approach → close gripper → lift

Usage
-----
  python3 scripts/grasp.py [--classes "can"] [--can can2] [--conf 0.25] [--vis]

Keys (during pipeline)
  q / Ctrl+C : abort at any time and return to home
"""

import os
import sys
import time
import argparse
import select
import tty
import termios

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

from config import GraspConfig
from utils import (
    build_fk, fk_T_base_ee, read_joints_deg,
    open_zed, sample_depth, backproject,
    read_gripper_feedback, close_gripper_until_grasp,
)
from ik_ee_ctrl import IKEndEffectorCtrl, CONTROL_DT


# ── Geometry ──────────────────────────────────────────────────────────────────

def horizontal_rotation(yaw_rad: float) -> np.ndarray:
    """
    Rotation matrix for a horizontal approach with level (non-rolling) jaws.

    Convention (right-hand, columns = axes):
      z-axis (approach)  = [cos θ, sin θ, 0]   horizontal, toward object
      y-axis (jaw plane) = [-sin θ, cos θ, 0]  horizontal, perpendicular
      x-axis             = [0, 0, -1]           vertical, pointing down

    Verify: x × y = z  →  [0,0,-1] × [-s,c,0] = [c, s, 0]  ✓
    """
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.column_stack([
        [ 0.0,  0.0, -1.0],   # x-axis (pointing down)
        [-s,    c,    0.0],   # y-axis (horizontal, perpendicular)
        [ c,    s,    0.0],   # z-axis (approach)
    ])


# ── Terminal helpers ─────────────────────────────────────────────────────────

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
    """Non-blocking check for q/Ctrl-C. Caller must already be in raw mode."""
    if select.select([sys.stdin], [], [], 0)[0]:
        ch = os.read(sys.stdin.fileno(), 4).decode("utf-8", errors="replace")
        return ch in ("q", "Q", "\x03")
    return False


def rprint(msg: str):
    """Print in raw-terminal mode (adds \\r before \\n)."""
    sys.stdout.write(msg.rstrip("\n") + "\r\n")
    sys.stdout.flush()


# ── Detection ─────────────────────────────────────────────────────────────────

def detect_frame(
    cam, rgb_mat, depth_mat, runtime, K,
    processor, det_model, device, text_prompt,
    piper, fk_model, fk_data, fk_ee_id, T_ee_cam,
    cfg: GraspConfig,
    vis_win: str | None = None,
) -> tuple[np.ndarray, float] | None:
    """
    Grab one ZED frame and run GroundingDINO.

    Returns (p_base, conf) for the highest-confidence detection above
    cfg.min_conf, or None if nothing found.
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
    labels = [results["labels"][i] for i, m in enumerate(mask) if m]

    joints = read_joints_deg(piper)
    T_be   = fk_T_base_ee(fk_model, fk_data, fk_ee_id, joints)

    best_p, best_conf = None, 0.0
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = y1 + (2.0 / 3.0) * (y2 - y1)   # 1/3 from bottom of bbox
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
    Accumulate n consecutive detections and return their median base-frame
    position. Resets on any miss. Returns None after miss_limit consecutive
    misses (object likely gone).
    """
    hits: list   = []
    misses: int  = 0
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

def sync_ctrl(ctrl: IKEndEffectorCtrl, sdk_values: list):
    """Force-sync IK controller internal state to given SDK joint values."""
    q = pin.neutral(ctrl.model)
    for i in range(6):
        q[i] = np.deg2rad(sdk_values[i] / 1000.0)
    q[6] = q[7] = 0.0
    ctrl.q = q
    pin.forwardKinematics(ctrl.model, ctrl.data, ctrl.q)
    pin.updateFramePlacements(ctrl.model, ctrl.data)
    T = ctrl.data.oMf[ctrl.ee_id]
    ctrl.target_pos = T.translation.copy()
    ctrl.target_rot = T.rotation.copy()


def smooth_to_joints(ctrl: IKEndEffectorCtrl, sdk_values: list, label: str = "moving"):
    """Smooth IK interpolation from current pose to the given SDK joint config."""
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
    """Smoothly: current → intermediate → zero; then open gripper."""
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
    ctrl.piper.GripperCtrl(ctrl.GRIPPER_OPEN_ANGLE, ctrl.GRIPPER_EFFORT, 0x01, 0)
    ctrl._gripper_open = True


# ── Stage 1 — Sweep until detected ───────────────────────────────────────────

def stage1_sweep(
    ctrl: IKEndEffectorCtrl,
    detect_fn,
    cfg: GraspConfig,
) -> np.ndarray | None:
    """
    Cycle through cfg.sweep_offsets_m around the current search pose.
    At each position, accumulate cfg.detection_frames consecutive detections.

    Returns the median base-frame object position, or None if q pressed.
    """
    center_pos      = ctrl.target_pos.copy()
    detections: list = []
    attempts_at_pos  = 0
    sweep_idx        = 0

    rprint("[Stage 1] Sweeping for target…  q = abort")

    while True:
        if quit_pressed():
            return None

        result = detect_fn()
        if result is not None:
            p, conf = result
            detections.append(p)
            attempts_at_pos = 0
            sys.stdout.write(
                f"\r  Detection {len(detections)}/{cfg.detection_frames}  "
                f"conf={conf:.2f}  "
                f"X={p[0]*100:+.1f} Y={p[1]*100:+.1f} Z={p[2]*100:+.1f} cm   "
            )
            sys.stdout.flush()
            if len(detections) >= cfg.detection_frames:
                median_p = np.median(np.array(detections), axis=0)
                rprint(
                    f"\n  Target locked: "
                    f"X={median_p[0]*100:+.2f} Y={median_p[1]*100:+.2f} "
                    f"Z={median_p[2]*100:+.2f} cm"
                )
                return median_p
        else:
            detections.clear()
            attempts_at_pos += 1
            sys.stdout.write(
                f"\r  No detection ({attempts_at_pos}/{cfg.scan_attempts_per_pos})   "
            )
            sys.stdout.flush()

            if attempts_at_pos >= cfg.scan_attempts_per_pos:
                attempts_at_pos = 0
                sweep_idx = (sweep_idx + 1) % len(cfg.sweep_offsets_m)
                offset = np.array(cfg.sweep_offsets_m[sweep_idx])
                dp = center_pos + offset - ctrl.target_pos
                rprint(
                    f"\n  Sweep: dX={offset[0]*100:+.1f} "
                    f"dY={offset[1]*100:+.1f} dZ={offset[2]*100:+.1f} cm"
                )
                ctrl._smooth_move(dp, np.eye(3), "sweep")


# ── Stage 2 — Align base yaw ─────────────────────────────────────────────────

def stage2_align_yaw(
    ctrl: IKEndEffectorCtrl,
    p_obj: np.ndarray,
    detect_fn,
    cfg: GraspConfig,
) -> np.ndarray | None:
    """
    Rotate joint1 to point toward the object, re-detect, repeat until yaw
    error < cfg.yaw_threshold_deg.

    Returns refined p_obj, or None if object lost (go back to Stage 1).
    """
    rprint("\n[Stage 2] Aligning yaw…")

    for iteration in range(cfg.yaw_max_iters):
        if quit_pressed():
            return None

        ideal_yaw_deg   = np.rad2deg(np.arctan2(p_obj[1], p_obj[0]))
        current_yaw_deg = np.rad2deg(ctrl.q[0])          # joint1 stored in radians
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

        rprint("  Re-detecting from new angle…")
        new_p = collect_detections(cfg.detection_frames, detect_fn)
        if new_p is None:
            rprint("  Object lost during yaw alignment.")
            return None
        p_obj = new_p

    rprint(f"  Yaw did not converge in {cfg.yaw_max_iters} iterations — proceeding.")
    return p_obj


# ── Stage 3 — Align EE height + gripper orientation ──────────────────────────

def stage3_align_ee(
    ctrl: IKEndEffectorCtrl,
    p_obj: np.ndarray,
    detect_fn,
    cfg: GraspConfig,
) -> np.ndarray | None:
    """
    Move link6 to the object's Z height and rotate the gripper to a horizontal
    orientation with level jaws (Y-axis of tool lies in the horizontal plane).

    Detection runs continuously to update the Z target as the arm moves.

    Returns latest p_obj when both height and orientation are aligned,
    or None on timeout / lost detection / q pressed.
    """
    rprint("\n[Stage 3] Aligning EE height and orientation…")

    STEPS_PER_DETECT = 5
    step_count       = 0
    deadline         = time.time() + cfg.align_timeout_s

    yaw_rad    = np.arctan2(p_obj[1], p_obj[0])
    target_rot = horizontal_rotation(yaw_rad)

    while time.time() < deadline:
        if quit_pressed():
            return None

        # # Re-detect to update height target every N steps
        # if step_count % STEPS_PER_DETECT == 0:
        #     result = detect_fn()
        #     if result is not None:
        #         p_obj      = result[0]
        #         yaw_rad    = np.arctan2(p_obj[1], p_obj[0])
        #         target_rot = horizontal_rotation(yaw_rad)

        step_count += 1

        target_z = p_obj[2]
        z_err    = target_z - ctrl.target_pos[2]

        dR_remaining = ctrl.target_rot.T @ target_rot
        omega        = pin.log3(dR_remaining)
        rot_err_deg  = np.rad2deg(np.linalg.norm(omega))

        sys.stdout.write(
            f"\r  dZ={z_err*100:+.1f}cm  rot_err={rot_err_deg:.1f}°   "
        )
        sys.stdout.flush()

        # Convergence check
        if abs(z_err) < cfg.height_threshold_m and rot_err_deg < cfg.orient_threshold_deg:
            rprint("\n  EE aligned.")
            return p_obj

        # Incremental IK step: Z adjustment + partial rotation
        dz = float(np.clip(z_err, -cfg.align_z_step_m, cfg.align_z_step_m))
        ctrl.target_pos = ctrl.target_pos + np.array([0.0, 0.0, dz])

        omega_norm = np.linalg.norm(omega)
        if omega_norm > 1e-6:
            ctrl.target_rot = ctrl.target_rot @ pin.exp3(cfg.align_rot_fraction * omega)

        joints_deg = ctrl._solve_ik()
        ctrl._send_joints(joints_deg)
        ctrl._print_state("align EE")
        time.sleep(CONTROL_DT)

    rprint("\n  Alignment timeout.")
    return None


# ── Stage 4 — Horizontal approach and grasp ───────────────────────────────────

def stage4_grasp(
    ctrl: IKEndEffectorCtrl,
    p_obj: np.ndarray,
    cfg: GraspConfig,
) -> bool:
    """
    Pure horizontal approach → close gripper with feedback → lift.

    The approach direction is computed from the final yaw to the object.
    The EE Z stays at the current (already aligned) height throughout.

    Returns True on successful grasp, False if aborted or no object detected.
    """
    rprint("\n[Stage 4] Approach and grasp…")

    yaw_rad    = np.arctan2(p_obj[1], p_obj[0])
    target_rot = horizontal_rotation(yaw_rad)

    # TCP offset: link6 → link7 fingertip in the target orientation
    tcp_mag         = np.linalg.norm(ctrl.get_tcp_offset_base())
    tcp_offset_vec  = target_rot[:, 2] * tcp_mag   # along tool Z-axis

    # Desired fingertip positions at current EE height (horizontal approach)
    grasp_tcp    = np.array([p_obj[0], p_obj[1], ctrl.target_pos[2]])
    pregrasp_tcp = grasp_tcp - target_rot[:, 2] * cfg.pregrasp_offset_m

    grasp_pos    = grasp_tcp    - tcp_offset_vec
    pregrasp_pos = pregrasp_tcp - tcp_offset_vec

    min_z              = ctrl.target_pos[2]   # do not descend during approach
    original_duration  = ctrl.move_duration
    ctrl.move_duration = cfg.approach_duration_s

    try:
        # 1. Open gripper
        rprint("[1/4] Opening gripper…")
        ctrl.piper.GripperCtrl(ctrl.GRIPPER_OPEN_ANGLE, ctrl.GRIPPER_EFFORT, 0x01, 0)
        ctrl._gripper_open = True
        time.sleep(0.5)
        if quit_pressed():
            return False

        # 2. Approach to pre-grasp (align remaining rotation + horizontal move)
        rprint("[2/4] Approaching pre-grasp…")
        dp = pregrasp_pos - ctrl.target_pos
        dR = ctrl.target_rot.T @ target_rot
        ctrl._smooth_move(dp, dR, "approach", min_z=min_z)
        if quit_pressed():
            return False

        # 3. Advance horizontally to grasp (no rotation change)
        rprint("[3/4] Advancing to grasp…")
        dp2 = grasp_pos - ctrl.target_pos
        ctrl._smooth_move(dp2, np.eye(3), "grasp", min_z=min_z)
        if quit_pressed():
            return False

        # 4. Close gripper with contact detection
        rprint("[4/4] Closing gripper…")
        grabbed, grip_fb = close_gripper_until_grasp(
            ctrl.piper,
            start_angle_raw=ctrl.GRIPPER_OPEN_ANGLE,
            effort=ctrl.GRIPPER_EFFORT,
            min_angle_raw=cfg.gripper_min_hold_angle_raw,
            step_angle_raw=cfg.gripper_close_step_raw,
            contact_effort_raw=cfg.gripper_contact_effort_raw,
            timeout=cfg.gripper_close_timeout_s,
            debug=True,
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
            f"\nGrasped! "
            f"angle={grip_fb['angle_mm']:.1f}mm  "
            f"effort={grip_fb['effort_nm']:+.2f}N·m"
        )

        # 5. Lift
        rprint("[5/5] Lifting…")
        ctrl._smooth_move(np.array([0.0, 0.0, cfg.lift_height_m]), np.eye(3), "lift")
        rprint("Grasp complete.")
        return True

    finally:
        ctrl.move_duration = original_duration


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Piper autonomous grasp pipeline")
    parser.add_argument("--classes", default=None,  help="Override config.classes")
    parser.add_argument("--can",     default=None,  help="Override config.can_port")
    parser.add_argument("--conf",    type=float, default=None,
                        help="Override config.min_conf")
    parser.add_argument("--vis",     action="store_true")
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

    # ── FK model (for detection coordinate transforms) ────────────────────────
    fk_model, fk_data, fk_ee_id = build_fk()

    # ── Piper arm ─────────────────────────────────────────────────────────────
    piper = C_PiperInterface_V2(cfg.can_port)
    piper.ConnectPort()
    time.sleep(0.2)

    ctrl = IKEndEffectorCtrl(piper, move_duration=2.0)
    ctrl.emergency_restore()
    ctrl.enable()
    ctrl.go_to_zero_and_sync()

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

    # ── Detection closure (captures all deps) ────────────────────────────────
    vis_win = "Grasp" if args.vis else None

    def detect():
        return detect_frame(
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

            # Enter raw-terminal mode for the entire pipeline
            orig = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            success = False
            try:
                # Stage 1: Sweep
                p_obj = stage1_sweep(ctrl, detect, cfg)
                if p_obj is None:
                    rprint("\nAborted during sweep.")
                else:
                    # Stage 2: Yaw alignment
                    p_obj = stage2_align_yaw(ctrl, p_obj, detect, cfg)
                    if p_obj is None:
                        rprint("\nTarget lost — will restart sweep.")
                    else:
                        # Safety: distance check
                        dist_m = float(np.linalg.norm(p_obj))
                        if dist_m > cfg.max_reach_m:
                            rprint(
                                f"\nObject too far "
                                f"({dist_m*100:.1f}cm > {cfg.max_reach_m*100:.0f}cm). "
                                "Restarting."
                            )
                        else:
                            # Stage 3: EE height + orientation
                            # p_obj = stage3_align_ee(ctrl, p_obj, detect, cfg)
                            if p_obj is None:
                                rprint("\nAlignment failed — restarting.")
                            else:
                                # Stage 4: Approach + grasp
                                success = stage4_grasp(ctrl, p_obj, cfg)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig)

            if success:
                print("\nGrasp succeeded!")
            else:
                print("\nReturning to home.")
            return_to_home(ctrl, cfg)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        cam.close()
        if args.vis:
            cv2.destroyAllWindows()
        print("Emergency stop.")
        try:
            ctrl.go_to_joints(list(cfg.intermediate_pose))
            ctrl.go_to_zero_and_sync()
        except Exception:
            pass
        ctrl.emergency_stop()


if __name__ == "__main__":
    main()
