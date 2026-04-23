#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Shared utilities for ZED-X + Piper arm scripts.

Exports
-------
  Constants : URDF_PATH, EE_FRAME, ARM_FACTOR
  FK        : build_fk, fk_T_base_ee
  Arm       : read_joints_deg
  ZED       : open_zed
  Geometry  : sample_depth, backproject
"""

import time

import numpy as np
import pinocchio as pin
import pyzed.sl as sl

# ── Shared constants ──────────────────────────────────────────────────────────
URDF_PATH  = "/workspace/piper_ros/src/piper_description/urdf/piper_description.urdf"
EE_FRAME   = "link6"
ARM_FACTOR = 1000   # SDK joint values = degrees × 1000


# ── FK helpers ────────────────────────────────────────────────────────────────

def build_fk(urdf_path: str = URDF_PATH, ee_frame: str = EE_FRAME):
    """Load URDF and return (model, data, ee_frame_id)."""
    model = pin.buildModelFromUrdf(urdf_path)
    data  = model.createData()
    ee_id = model.getFrameId(ee_frame)
    return model, data, ee_id


def fk_T_base_ee(model, data, ee_id, joint_angles_deg: list) -> np.ndarray:
    """Return 4×4 homogeneous T_base_ee from 6 joint angles in degrees."""
    q = pin.neutral(model)
    for i, deg in enumerate(joint_angles_deg):
        q[i] = np.deg2rad(deg)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T = data.oMf[ee_id]
    M = np.eye(4)
    M[:3, :3] = T.rotation
    M[:3,  3] = T.translation
    return M


# ── Arm helpers ───────────────────────────────────────────────────────────────

def read_joints_deg(piper) -> list:
    """Read current joint angles from the Piper SDK (returns degrees)."""
    fb = piper.GetArmJointMsgs().joint_state
    return [
        fb.joint_1 / ARM_FACTOR,
        fb.joint_2 / ARM_FACTOR,
        fb.joint_3 / ARM_FACTOR,
        fb.joint_4 / ARM_FACTOR,
        fb.joint_5 / ARM_FACTOR,
        fb.joint_6 / ARM_FACTOR,
    ]


def _print_ee_pose(piper) -> None:
    """Print the current EE pose from the arm's built-in FK."""
    try:
        ep = piper.GetArmEndPoseMsgs().end_pose
        x  = ep.X_axis  / 1000.0
        y  = ep.Y_axis  / 1000.0
        z  = ep.Z_axis  / 1000.0
        rx = ep.RX_axis / 1000.0
        ry = ep.RY_axis / 1000.0
        rz = ep.RZ_axis / 1000.0
        print(
            f"link6  X={x:+.1f}mm  Y={y:+.1f}mm  Z={z:+.1f}mm  "
            f"RX={rx:+.1f}°  RY={ry:+.1f}°  RZ={rz:+.1f}°"
        )
    except Exception:
        pass


def go_to_joints(piper, values: list, speed: int = 50, tolerance: int = 500, timeout: float = 10.0, verbose: bool = True):
    """
    Move the arm to the given SDK joint values (degrees × 1000) and wait until reached.
    
    Parameters
    ----------
    piper : C_PiperInterface_V2
        Piper SDK interface
    values : list
        6 joint angles in SDK units (degrees × 1000)
    speed : int
        Motion speed percentage (0-100)
    tolerance : int
        Position tolerance in SDK units (default 500 = 0.5°)
    timeout : float
        Maximum time to wait in seconds
    verbose : bool
        Print progress messages
    """
    if verbose:
        print(f"Moving to joints {values}...")
    start = time.time()
    while time.time() - start < timeout:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*values)
        fb = piper.GetArmJointMsgs().joint_state
        angles = [
            fb.joint_1, fb.joint_2, fb.joint_3,
            fb.joint_4, fb.joint_5, fb.joint_6,
        ]
        if all(abs(angles[i] - values[i]) < tolerance for i in range(6)):
            if verbose:
                print("Target position reached.")
                _print_ee_pose(piper)
            return True
        time.sleep(0.05)
    if verbose:
        print("Target position timeout, continuing anyway.")
        _print_ee_pose(piper)
    return False


def read_gripper_feedback(piper) -> dict:
    """Read gripper angle, effort, and status flags from the Piper SDK."""
    msg = piper.GetArmGripperMsgs()
    state = getattr(msg, "gripper_state", msg)
    status = getattr(state, "status_code", None)
    if status is None:
        status = getattr(msg, "status_code", None)

    angle_raw = int(getattr(state, "grippers_angle", getattr(msg, "grippers_angle", 0)))
    effort_raw = int(getattr(state, "grippers_effort", getattr(msg, "grippers_effort", 0)))

    driver_enabled = None
    if status is not None:
        driver_enabled = bool(getattr(status, "driver_enable_status", False))
    fault_flags = {
        "voltage_too_low": bool(getattr(status, "voltage_too_low", False)) if status is not None else False,
        "motor_overheating": bool(getattr(status, "motor_overheating", False)) if status is not None else False,
        "driver_overcurrent": bool(getattr(status, "driver_overcurrent", False)) if status is not None else False,
        "driver_overheating": bool(getattr(status, "driver_overheating", False)) if status is not None else False,
        "sensor_status": bool(getattr(status, "sensor_status", False)) if status is not None else False,
        "driver_error_status": bool(getattr(status, "driver_error_status", False)) if status is not None else False,
    }

    return {
        "angle_raw": angle_raw,
        "angle_mm": angle_raw / 1000.0,
        "angle_m": angle_raw / 1_000_000.0,
        "effort_raw": effort_raw,
        "effort_nm": effort_raw / 1000.0,
        "driver_enabled": driver_enabled,
        "status_available": status is not None,
        "faulted": any(fault_flags.values()),
        "fault_flags": fault_flags,
    }


def gripper_has_object(
    feedback: dict,
    effort_threshold_raw: int = 120,
    min_contact_angle_raw: int = 3000,
) -> bool:
    """Return True when the gripper likely contacted and is holding an object."""
    return (
        not feedback["faulted"]
        and abs(feedback["effort_raw"]) >= effort_threshold_raw
        and feedback["angle_raw"] > min_contact_angle_raw
    )


def close_gripper_until_grasp(
    piper,
    start_angle_raw: int,
    effort: int,
    min_angle_raw: int = 3000,
    step_angle_raw: int = 4000,
    contact_effort_raw: int = 120,
    settle_time: float = 0.08,
    timeout: float = 3.0,
    angle_tolerance_raw: int = 1500,
    debug: bool = False,
) -> tuple[bool, dict]:
    """
    Close the gripper in small steps and stop early once contact is detected.

    Returns `(grabbed, feedback)`, where `grabbed` is True when the effort and
    angle feedback suggest an object was contacted before the gripper reached
    the minimum allowed closing angle.
    """
    deadline = time.time() + timeout
    command_angle = int(start_angle_raw)
    last_feedback = read_gripper_feedback(piper)
    
    # Track baseline effort to detect rising resistance during closure
    baseline_effort = abs(last_feedback['effort_raw'])
    effort_rise_threshold = max(contact_effort_raw, baseline_effort + 200)
    
    if debug:
        print(f"\n[GRIPPER] Starting close: start={start_angle_raw} min={min_angle_raw} "
              f"step={step_angle_raw} baseline_effort={baseline_effort} "
              f"rise_thresh={effort_rise_threshold} timeout={timeout:.1f}s")
        print(f"[GRIPPER] Initial feedback: angle={last_feedback['angle_raw']} "
              f"effort={last_feedback['effort_raw']} enabled={last_feedback['driver_enabled']} "
              f"faulted={last_feedback['faulted']}")

    step_count = 0
    while time.time() < deadline:
        piper.GripperCtrl(command_angle, effort, 0x01, 0)
        time.sleep(settle_time)
        last_feedback = read_gripper_feedback(piper)
        step_count += 1

        if debug and step_count % 5 == 0:
            print(f"[GRIPPER] Step {step_count}: cmd={command_angle} "
                  f"fb_angle={last_feedback['angle_raw']} fb_effort={last_feedback['effort_raw']} "
                  f"enabled={last_feedback['driver_enabled']} faulted={last_feedback['faulted']}")

        # Check for contact: effort has risen AND gripper has closed from start
        current_effort = abs(last_feedback['effort_raw'])
        has_closed = last_feedback['angle_raw'] < (start_angle_raw - step_angle_raw)
        
        if (not last_feedback["faulted"] 
            and has_closed
            and current_effort >= effort_rise_threshold
            and last_feedback["angle_raw"] > min_angle_raw):
            if debug:
                print(f"[GRIPPER] ✓ OBJECT DETECTED at step {step_count}: "
                      f"angle={last_feedback['angle_raw']} effort={current_effort} "
                      f"(rose from {baseline_effort})")
            return True, last_feedback
        
        if last_feedback["faulted"]:
            if debug:
                print(f"[GRIPPER] ✗ EXIT: fault detected at step {step_count}: {last_feedback['fault_flags']}")
            return False, last_feedback

        # Only advance to the next closing step after the real gripper angle has
        # nearly reached the last command. This keeps timeout meaningful even
        # when the hardware reports feedback more slowly than the command loop.
        if command_angle > min_angle_raw and last_feedback["angle_raw"] <= command_angle + angle_tolerance_raw:
            old_cmd = command_angle
            command_angle = max(min_angle_raw, command_angle - step_angle_raw)
            if debug and old_cmd != command_angle:
                print(f"[GRIPPER] Stepping down: {old_cmd} → {command_angle} "
                      f"(fb_angle={last_feedback['angle_raw']} within tolerance)")

    if debug:
        elapsed = timeout
        print(f"[GRIPPER] ⏱ TIMEOUT after {elapsed:.1f}s ({step_count} steps)")
    
    piper.GripperCtrl(command_angle, effort, 0x01, 0)
    time.sleep(settle_time)
    last_feedback = read_gripper_feedback(piper)
    
    # Final check with same logic
    current_effort = abs(last_feedback['effort_raw'])
    has_closed = last_feedback['angle_raw'] < (start_angle_raw - step_angle_raw)
    grabbed = (not last_feedback["faulted"]
               and has_closed
               and current_effort >= effort_rise_threshold
               and last_feedback["angle_raw"] > min_angle_raw)
    
    if debug:
        print(f"[GRIPPER] Final check: grabbed={grabbed} "
              f"angle={last_feedback['angle_raw']} effort={current_effort} baseline={baseline_effort}")
    
    return grabbed, last_feedback


# ── ZED helpers ───────────────────────────────────────────────────────────────

def open_zed(
    depth_mode=sl.DEPTH_MODE.NEURAL,
    resolution=sl.RESOLUTION.HD1200,
    depth_min_m: float = 0.05,
    depth_max_m: float = 3.0,
):
    """
    Open the ZED camera and return (cam, rgb_mat, depth_mat, runtime, K, dist).

    Parameters
    ----------
    depth_mode   : sl.DEPTH_MODE  — use NONE to skip depth (e.g. during calibration)
    resolution   : sl.RESOLUTION
    depth_min_m  : float  — minimum depth clamp (metres)
    depth_max_m  : float  — maximum depth clamp (metres)

    Returns
    -------
    cam        : sl.Camera
    rgb_mat    : sl.Mat  (left view)
    depth_mat  : sl.Mat  (DEPTH measure)
    runtime    : sl.RuntimeParameters
    K          : np.ndarray (3×3) left camera intrinsic matrix
    dist       : np.ndarray (N,)  distortion coefficients
    """
    cam  = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution      = resolution
    init.depth_mode             = depth_mode
    init.coordinate_units       = sl.UNIT.METER
    init.depth_minimum_distance = depth_min_m
    init.depth_maximum_distance = depth_max_m

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")

    cal  = cam.get_camera_information().camera_configuration.calibration_parameters.left_cam
    K    = np.array([[cal.fx,    0, cal.cx],
                     [   0, cal.fy, cal.cy],
                     [   0,      0,      1]], dtype=np.float64)
    dist = np.array(cal.disto, dtype=np.float64)

    rgb_mat   = sl.Mat()
    depth_mat = sl.Mat()
    runtime   = sl.RuntimeParameters()
    return cam, rgb_mat, depth_mat, runtime, K, dist


# ── Geometry helpers ──────────────────────────────────────────────────────────

def sample_depth(
    depth_arr: np.ndarray,
    cx: float,
    cy: float,
    patch: int = 5,
    depth_min_m: float = 0.05,
    depth_max_m: float = 3.0,
) -> float | None:
    """
    Return the median depth (metres) in a (2*patch+1)² patch around (cx, cy).
    Returns None if no valid pixels are found.
    """
    h, w = depth_arr.shape
    x0 = max(0, int(cx) - patch)
    x1 = min(w, int(cx) + patch + 1)
    y0 = max(0, int(cy) - patch)
    y1 = min(h, int(cy) + patch + 1)
    region = depth_arr[y0:y1, x0:x1].flatten()
    valid  = region[np.isfinite(region) & (region > depth_min_m) & (region < depth_max_m)]
    if len(valid) == 0:
        return None
    return float(np.median(valid))


def backproject(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """
    Back-project a pixel (u, v) at known depth into the camera frame.

    Returns a (3,) array [X, Y, Z] in metres.
    """
    x = (u - K[0, 2]) * depth_m / K[0, 0]
    y = (v - K[1, 2]) * depth_m / K[1, 1]
    return np.array([x, y, depth_m])
