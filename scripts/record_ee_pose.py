#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Live end-effector pose display + recorder for the Piper arm.

  Display: updates in-place every ~50 ms
  Space  : record current pose (with timestamp) to the output file
  q / Ctrl+C : quit

Usage:
    python record_ee_pose.py [--can can2] [--out recorded_poses.txt]
"""

import sys
import os
import tty
import termios
import select
import time
import argparse
from datetime import datetime

import numpy as np
import pinocchio as pin

from piper_sdk import C_PiperInterface_V2

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_CAN   = "can2"
DEFAULT_OUT   = "/workspace/scripts/recorded_poses.txt"
POLL_HZ        = 20
POLL_DT        = 1.0 / POLL_HZ
URDF_PATH      = "/workspace/piper_ros/src/piper_description/urdf/piper_description.urdf"
FINGER1_FRAME  = "link7"
FINGER2_FRAME  = "link8"
ARM_FACTOR     = 1000   # SDK joint values = degrees × 1000


def _build_model():
    model = pin.buildModelFromUrdf(URDF_PATH)
    data  = model.createData()
    f1_id = model.getFrameId(FINGER1_FRAME)
    f2_id = model.getFrameId(FINGER2_FRAME)
    return model, data, f1_id, f2_id


# ── Helpers ───────────────────────────────────────────────────────────────────
def read_pose(piper: C_PiperInterface_V2, model, data, f1_id, f2_id):
    """Return (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg) of the finger midpoint, or None."""
    try:
        fb = piper.GetArmJointMsgs().joint_state
        q  = pin.neutral(model)
        for i, val in enumerate([
            fb.joint_1, fb.joint_2, fb.joint_3,
            fb.joint_4, fb.joint_5, fb.joint_6,
        ]):
            q[i] = np.deg2rad(val / ARM_FACTOR)
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        midpoint_m = 0.5 * (
            data.oMf[f1_id].translation + data.oMf[f2_id].translation
        )
        ep = piper.GetArmEndPoseMsgs().end_pose
        rx = ep.RX_axis / 1000.0
        ry = ep.RY_axis / 1000.0
        rz = ep.RZ_axis / 1000.0
        x, y, z = midpoint_m * 1000.0
        return x, y, z, rx, ry, rz
    except Exception:
        return None


def pose_display(pose):
    """Format pose as a compact display string."""
    x, y, z, rx, ry, rz = pose
    return (
        f"midpt  "
        f"X={x:+.1f}mm  "
        f"Y={y:+.1f}mm  "
        f"Z={z:+.1f}mm  "
        f"RX={rx:+.1f}°  "
        f"RY={ry:+.1f}°  "
        f"RZ={rz:+.1f}°"
    )


def pose_record_line(pose, index: int):
    """Format a timestamped line for the output file."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    x, y, z, rx, ry, rz = pose
    return (
        f"[{index:04d}] {ts}  "
        f"X={x:+.3f}mm  "
        f"Y={y:+.3f}mm  "
        f"Z={z:+.3f}mm  "
        f"RX={rx:+.3f}°  "
        f"RY={ry:+.3f}°  "
        f"RZ={rz:+.3f}°\n"
    )


# ── Raw keyboard ──────────────────────────────────────────────────────────────
def set_raw(fd):
    old = termios.tcgetattr(fd)
    tty.setraw(fd)
    return old


def restore_terminal(fd, old):
    termios.tcsetattr(fd, termios.TCSADRAIN, old)


def key_pressed(fd, timeout=0.0):
    """Return the pressed character, or None if nothing within timeout."""
    rlist, _, _ = select.select([fd], [], [], timeout)
    if rlist:
        ch = os.read(fd, 4)
        return ch
    return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live EE pose viewer & recorder")
    parser.add_argument("--can", default=DEFAULT_CAN, help="CAN interface name")
    parser.add_argument("--out", default=DEFAULT_OUT,  help="Output file path")
    args = parser.parse_args()

    print(f"Connecting to arm on {args.can} …")
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    time.sleep(0.5)

    model, data, f1_id, f2_id = _build_model()

    print(f"Recording to: {args.out}")
    print("Controls: [Space] record pose   [q / Ctrl+C] quit\n")

    fd  = sys.stdin.fileno()
    old = set_raw(fd)

    record_count = 0
    last_line_len = 0
    status_msg = ""        # shown after the pose line on record

    try:
        while True:
            t0 = time.monotonic()

            # ── Read key (non-blocking) ────────────────────────────────────
            ch = key_pressed(fd, timeout=0.0)
            if ch is not None:
                if ch in (b'q', b'Q', b'\x03'):   # q or Ctrl+C
                    break
                elif ch == b' ':                   # Space → record
                    pose = read_pose(piper, model, data, f1_id, f2_id)
                    if pose is not None:
                        record_count += 1
                        line = pose_record_line(pose, record_count)
                        with open(args.out, "a") as f:
                            f.write(line)
                        status_msg = f"  ← #{record_count} saved"
                    else:
                        status_msg = "  ← read error"

            # ── Refresh display ────────────────────────────────────────────
            pose = read_pose(piper, model, data, f1_id, f2_id)
            if pose is not None:
                disp = pose_display(pose) + status_msg
            else:
                disp = "[waiting for arm data…]" + status_msg

            # Overwrite previous line in-place
            clear = "\r" + " " * last_line_len + "\r"
            sys.stdout.write(clear + disp)
            sys.stdout.flush()
            last_line_len = len(disp)

            # ── Sleep for remainder of poll period ─────────────────────────
            elapsed = time.monotonic() - t0
            sleep_t = max(0.0, POLL_DT - elapsed)
            time.sleep(sleep_t)

    finally:
        restore_terminal(fd, old)
        sys.stdout.write("\n")
        sys.stdout.flush()
        print(f"Done. {record_count} pose(s) saved to {args.out}")


if __name__ == "__main__":
    main()
