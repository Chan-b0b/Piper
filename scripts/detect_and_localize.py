#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Detect an object with YOLOWorld (open-vocabulary) using the ZED-X camera,
localise it in 3-D using the ZED depth map, and transform the point into
the robot base frame using the hand-eye calibration result.

Usage
-----
  python3 scripts/detect_and_localize.py \
      --classes "red cup" \
      --can can2 \
      [--calib  /workspace/calibration/T_ee_cam.npy] \
      [--conf   0.25] \
      [--vis]

Keys (terminal)
  q / Ctrl+C  : quit
"""

import os
import sys
import argparse
import select
import tty
import termios
import time

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
    open_zed, sample_depth, backproject,
)

# YOLOWorld — imported after env vars are set so CUDA init is correct
from ultralytics import YOLOWorld

# ── Constants ──────────────────────────────────────────────────────────────────
DEPTH_PATCH = 5      # half-size of patch around bbox centre used for depth median (pixels)
DEPTH_MIN_M = 0.05
DEPTH_MAX_M = 3.0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect & localise object with ZED + YOLOWorld")
    parser.add_argument("--classes", default="A crumpled ball of white paper",
                        help="Comma-separated object class names (e.g. 'red cup,bottle')")
    parser.add_argument("--can",    default="can2")
    parser.add_argument("--calib",  default="/workspace/calibration/T_ee_cam.npy")
    parser.add_argument("--model",  default="yolov8s-worldv2.pt")
    parser.add_argument("--conf",   type=float, default=0.25)
    parser.add_argument("--vis",    action="store_true", help="Show OpenCV window")
    args = parser.parse_args()

    class_list = [c.strip() for c in args.classes.split(",")]
    print(f"Detecting: {class_list}")

    # ── Load calibration ──────────────────────────────────────────────────────
    if not os.path.exists(args.calib):
        print(f"ERROR: calibration file not found: {args.calib}")
        sys.exit(1)
    T_ee_cam = np.load(args.calib)
    print(f"T_ee_cam loaded from {args.calib}")

    # ── Pinocchio FK ──────────────────────────────────────────────────────────
    model, data, ee_id = build_fk(URDF_PATH, EE_FRAME)

    # ── Piper SDK ─────────────────────────────────────────────────────────────
    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    time.sleep(0.2)

    # ── ZED camera ────────────────────────────────────────────────────────────
    print("Opening ZED camera...")
    cam, rgb_mat, depth_mat, runtime, K, _ = open_zed(
        depth_mode=sl.DEPTH_MODE.NEURAL,
        depth_min_m=DEPTH_MIN_M,
        depth_max_m=DEPTH_MAX_M,
    )
    print("ZED opened.")

    # ── YOLOWorld model ───────────────────────────────────────────────────────
    print(f"Loading YOLOWorld model: {args.model}")
    detector = YOLOWorld(args.model)
    detector.set_classes(class_list)
    print("Model ready.")

    # ── Terminal raw input ────────────────────────────────────────────────────
    orig_settings = termios.tcgetattr(sys.stdin)
    tty.setraw(sys.stdin.fileno())

    print("\nRunning… press q to quit.\n")

    best_p_base = None
    try:
        while True:
            # ── Non-blocking key check ────────────────────────────────────────
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key in ('q', 'Q', '\x03'):
                    break

            # ── Grab frame ───────────────────────────────────────────────────
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            cam.retrieve_image(rgb_mat,   sl.VIEW.LEFT)
            cam.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

            frame = np.ascontiguousarray(rgb_mat.get_data()[:, :, :3])   # BGR, no alpha; force standard numpy type
            depth = depth_mat.get_data()                   # float32 metres

            # ── YOLOWorld inference ───────────────────────────────────────────
            results = detector.predict(frame, conf=args.conf, verbose=False)
            boxes   = results[0].boxes

            vis = frame.copy() if args.vis else None

            best_p_base = None
            best_conf   = 0.0

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val        = float(box.conf[0])
                cls_id          = int(box.cls[0])
                label           = class_list[cls_id] if cls_id < len(class_list) else str(cls_id)

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                depth_m = sample_depth(depth, cx, cy, patch=DEPTH_PATCH,
                                        depth_min_m=DEPTH_MIN_M, depth_max_m=DEPTH_MAX_M)
                if depth_m is None:
                    sys.stdout.write(f"\r  [{label}] conf={conf_val:.2f}  depth=INVALID           ")
                    sys.stdout.flush()
                    continue

                # 3-D in camera frame
                p_cam = backproject(cx, cy, depth_m, K)

                # FK: T_base_ee from current joint positions
                joints_deg = read_joints_deg(piper)
                T_base_ee  = fk_T_base_ee(model, data, ee_id, joints_deg)

                # Transform: camera → EE → base
                p_base = (T_base_ee @ T_ee_cam @ np.append(p_cam, 1.0))[:3]

                sys.stdout.write(
                    f"\r  [{label}] conf={conf_val:.2f}  "
                    f"depth={depth_m*100:.1f}cm  "
                    f"base: X={p_base[0]*100:+.1f}cm Y={p_base[1]*100:+.1f}cm Z={p_base[2]*100:+.1f}cm   "
                )
                sys.stdout.flush()

                if conf_val > best_conf:
                    best_conf   = conf_val
                    best_p_base = p_base

                if vis is not None:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(vis,
                                f"{label} {conf_val:.2f} | d={depth_m*100:.1f}cm",
                                (int(x1), int(y1) - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

            if args.vis and vis is not None:
                cv2.imshow("ZED + YOLOWorld", vis)
                cv2.waitKey(1)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)
        cam.close()
        if args.vis:
            cv2.destroyAllWindows()
        print("\nDone.")
        if best_p_base is not None:
            print(f"\nLast best detection in base frame:")
            print(f"  X={best_p_base[0]*100:+.2f} cm")
            print(f"  Y={best_p_base[1]*100:+.2f} cm")
            print(f"  Z={best_p_base[2]*100:+.2f} cm")


if __name__ == "__main__":
    main()


#python3 detect_and_localize.py --classes "plastic cup" --can can2 --vis

#pip install "git+https://github.com/pytorch/vision.git@v0.20.1"