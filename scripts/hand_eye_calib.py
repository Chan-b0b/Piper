#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Hand-eye calibration for ZED-X (eye-in-hand) on the Piper arm.

Usage
-----
  python3 scripts/hand_eye_calib.py [--can can2] [--marker-size 0.10] [--out T_ee_cam.npy]

Procedure
---------
  1. Print calibration/aruco_marker.png (100 mm ArUco target) and fix it
     rigidly somewhere in the robot's workspace.
  2. Run this script — the ZED view appears.
  3. Move the arm with your preferred method (e.g. teach-pendant / ik_ee_ctrl)
     so the marker fills roughly 1/3 of the frame.
  4. Press SPACE  → captures current pose pair (T_base_ee + T_cam_marker).
     Press D       → delete the last capture.
     Repeat for ≥ 15 diverse poses (vary both position AND orientation).
  5. Press C       → solve hand-eye calibration (needs ≥ 4 captures).
  6. Press S       → save result to --out file.
  7. Press Q       → quit.

Output
------
  A (4×4) numpy array saved as `T_ee_cam.npy`:
    T_ee_cam  — rigid transform from end-effector (link6) frame to
                ZED left-camera frame.
  Apply as:
    p_base = T_base_ee  @ T_ee_cam  @ p_cam_hom
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
import pyzed.sl as sl
from piper_sdk import C_PiperInterface_V2
from utils import (
    URDF_PATH, EE_FRAME, ARM_FACTOR,
    build_fk, fk_T_base_ee, read_joints_deg,
    open_zed,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ARUCO_DICT_ID = cv2.aruco.DICT_6X6_250
MARKER_ID     = 0           # which marker ID to track


# ── ArUco helper ───────────────────────────────────────────────────────────────

def detect_marker(frame_bgr, K, dist, marker_size_m: float):
    """
    Returns (rvec, tvec) in camera frame if MARKER_ID is found, else None.
    Uses solvePnP directly (cv2.aruco.estimatePoseSingleMarkers removed in OpenCV 4.8+).
    """
    aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None, None, frame_bgr

    vis = frame_bgr.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)

    # 3-D corners of the marker in its own frame (centre at origin, Z forward)
    h = marker_size_m / 2.0
    obj_pts = np.array([[-h,  h, 0],
                        [ h,  h, 0],
                        [ h, -h, 0],
                        [-h, -h, 0]], dtype=np.float32)

    for idx, mid in enumerate(ids.flatten()):
        if mid == MARKER_ID:
            img_pts = corners[idx].reshape(4, 2).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist,
                                           flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                cv2.drawFrameAxes(vis, K, dist, rvec, tvec, marker_size_m * 0.5)
                return rvec.squeeze(), tvec.squeeze(), vis

    return None, None, vis


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    T    = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec
    return T


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZED-X / Piper hand-eye calibration")
    parser.add_argument("--can",         default="can2",       help="CAN port")
    parser.add_argument("--marker-size", type=float, default=0.10,
                        help="ArUco marker physical side length in metres (default: 0.10)")
    parser.add_argument("--out",         default="/workspace/calibration/T_ee_cam.npy",
                        help="Output path for the 4×4 T_ee_cam transform")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # ── Pinocchio FK ──────────────────────────────────────────────────────────
    print("Loading URDF for FK...")
    model, data, ee_id = build_fk(URDF_PATH, EE_FRAME)

    # ── Piper SDK ─────────────────────────────────────────────────────────────
    print(f"Connecting to CAN port {args.can}...")
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    time.sleep(0.2)

    # ── ZED camera ────────────────────────────────────────────────────────────
    print("Opening ZED camera...")
    cam, img_mat, _, runtime, K, dist = open_zed(depth_mode=sl.DEPTH_MODE.NONE)
    print(f"  K  =\n{K}")
    print(f"  dist = {dist}")

    # ── Capture buffers ───────────────────────────────────────────────────────
    R_base_ee_list   = []   # list of 3×3
    t_base_ee_list   = []   # list of (3,)
    R_cam_marker_list = []
    t_cam_marker_list = []

    T_ee_cam = None   # result once solved

    print("\nControls:  SPACE=capture  D=delete last  C=solve  S=save  Q=quit\n")

    while True:
        if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
            continue

        cam.retrieve_image(img_mat, sl.VIEW.LEFT)
        frame = img_mat.get_data()[:, :, :3].copy()

        rvec, tvec, vis = detect_marker(frame, K, dist, args.marker_size)

        n = len(R_base_ee_list)

        # ── Overlay status ────────────────────────────────────────────────────
        marker_status = "MARKER FOUND" if rvec is not None else "no marker"
        color         = (0, 255, 0) if rvec is not None else (0, 0, 255)
        cv2.putText(vis, f"Captures: {n}   [{marker_status}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        if T_ee_cam is not None:
            cv2.putText(vis, "SOLVED — press S to save",
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Hand-Eye Calibration", vis)
        key = cv2.waitKey(1) & 0xFF

        # ── SPACE: capture ────────────────────────────────────────────────────
        if key == ord(' '):
            if rvec is None:
                print("  [SKIP] Marker not detected — move so the marker is visible.")
            else:
                joints_deg = read_joints_deg(piper)
                T_be       = fk_T_base_ee(model, data, ee_id, joints_deg)
                T_cm       = rvec_tvec_to_T(rvec, tvec)

                R_base_ee_list.append(T_be[:3, :3])
                t_base_ee_list.append(T_be[:3,  3])
                R_cam_marker_list.append(T_cm[:3, :3])
                t_cam_marker_list.append(T_cm[:3,  3])

                print(
                    f"  [CAPTURED #{n+1}]  joints={[f'{d:.1f}' for d in joints_deg]}"
                    f"  tvec_cam={tvec.round(4)}"
                )

        # ── D: delete last ─────────────────────────────────────────────────────
        elif key == ord('d') and n > 0:
            R_base_ee_list.pop()
            t_base_ee_list.pop()
            R_cam_marker_list.pop()
            t_cam_marker_list.pop()
            print(f"  [DELETED]  captures remaining: {n - 1}")

        # ── C: solve ──────────────────────────────────────────────────────────
        elif key == ord('c'):
            if n < 4:
                print(f"  [SKIP] Need at least 4 captures (have {n}).")
            else:
                print(f"\n  Solving hand-eye with {n} capture(s)...")
                R_ec, t_ec = cv2.calibrateHandEye(
                    R_base_ee_list,    t_base_ee_list,
                    R_cam_marker_list, t_cam_marker_list,
                    method=cv2.CALIB_HAND_EYE_TSAI,
                )
                T_ee_cam       = np.eye(4)
                T_ee_cam[:3, :3] = R_ec
                T_ee_cam[:3,  3] = t_ec.squeeze()
                print(f"  T_ee_cam =\n{T_ee_cam.round(6)}")

                # ── Consistency check: marker should map to same base-frame point ──
                # For each pose: p_base = T_base_ee @ T_ee_cam @ [t_cam_marker; 1]
                # All poses should agree → low spread = good calibration.
                p_base_list = []
                for R_be, t_be, R_cm, t_cm in zip(
                    R_base_ee_list, t_base_ee_list,
                    R_cam_marker_list, t_cam_marker_list,
                ):
                    T_be_m = np.eye(4); T_be_m[:3, :3] = R_be; T_be_m[:3, 3] = t_be
                    p_base = (T_be_m @ T_ee_cam @ np.append(t_cm, 1.0))[:3]
                    p_base_list.append(p_base)
                p_arr  = np.array(p_base_list)
                spread = np.linalg.norm(p_arr - p_arr.mean(axis=0), axis=1)
                rms    = np.sqrt(np.mean(spread ** 2))
                print(f"  Marker position consistency RMS: {rms*1000:.2f} mm  (lower is better)")

        # ── S: save ───────────────────────────────────────────────────────────
        elif key == ord('s'):
            if T_ee_cam is None:
                print("  [SKIP] Solve first (press C).")
            else:
                np.save(args.out, T_ee_cam)
                print(f"  [SAVED] T_ee_cam → {args.out}")

        # ── Q: quit ───────────────────────────────────────────────────────────
        elif key == ord('q'):
            break

    cam.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
