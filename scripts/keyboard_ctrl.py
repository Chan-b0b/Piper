#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Keyboard-controlled joint control for the Piper arm.

    Up   / Down  : Joint 2  +/- step
    Left / Right : Joint 3  +/- step
    m    / n     : Joint 5  +/- step
    c            : Toggle gripper open (70°) / closed (0°)
    +    / -     : Increase / decrease step size (5° increments, range 1–45°)
    q or Ctrl+C  : Quit (triggers emergency stop)

Startup : emergency restore → read current joint angles → enable
Shutdown: emergency stop
"""

import sys
import os
import tty
import termios
import select
import time
import argparse
from piper_sdk import C_PiperInterface

# ── Joint limits (degrees) ───────────────────────────────────────────────────
JOINT_LIMITS = [
    (-154.0, 154.0),  # Joint 1 — Base
    (   0.0, 195.0),  # Joint 2
    (-175.0,   0.0),  # Joint 3
    (-100.0, 112.0),  # Joint 4
    ( -75.0,  75.0),  # Joint 5
    (-120.0, 120.0),  # Joint 6
]

FACTOR = 1000   # SDK expects degrees × 1000 as integers
SPEED  = 50     # Motion speed percentage (0–100)


class KeyboardJointCtrl:
    GRIPPER_OPEN_ANGLE   = 70000  # 70° × 1000 sent to SDK
    GRIPPER_CLOSED_ANGLE = 0      # 0° = closed
    GRIPPER_EFFORT       = 1000   # 1 N/m

    def __init__(self, piper: C_PiperInterface, step: float = 10.0):
        self.piper          = piper
        self.step           = step
        self.joints         = [0.0] * 6   # current target angles (degrees)
        self._orig_settings = None
        self._estop_sent    = False
        self._gripper_open  = False        # tracks current gripper state

    # ── Emergency stop / restore ──────────────────────────────────────────────
    def emergency_stop(self):
        """Send E-stop command. Safe to call multiple times."""
        if not self._estop_sent:
            self._estop_sent = True
            print("\nSending emergency stop...")
            self.piper.MotionCtrl_1(0x01, 0, 0x00)

    def emergency_restore(self):
        """Restore from E-stop and switch the arm to CAN control mode."""
        print("Running emergency restore...")
        self.piper.MotionCtrl_1(0x02, 0, 0x00)
        self.piper.MotionCtrl_1(0x00, 0, 0x00)

        self.piper.MotionCtrl_2(0x01, 0, 0, 0x00)  # StandBy mode
        self.piper.GripperCtrl(0, 0, 0x02, 0)
        time.sleep(1)

        self.piper.MotionCtrl_2(0x01, 0, 0, 0x00)  # CAN mode
        self.piper.GripperCtrl(0, 0, 0x03, 0)
        time.sleep(1)

        if self.piper.GetArmStatus().arm_status.ctrl_mode == 0x01:
            print("Restore successful — arm is ready.")
        else:
            print("Restore failed — do not operate the arm!")
            sys.exit(1)

    # ── Arm enable ────────────────────────────────────────────────────────────
    def enable(self, timeout: float = 5.0):
        """Activate the arm and wait for all motors to become ready."""
        start = time.time()
        while time.time() - start < timeout:
            self.piper.EnableArm(7)
            self.piper.GripperCtrl(0, 1000, 0x01, 0)
            if all(
                getattr(self.piper.GetArmLowSpdInfoMsgs(), f"motor_{i}").foc_status.driver_enable_status
                for i in range(1, 7)
            ):
                print("Arm enabled.")
                return
            time.sleep(1)
        print("Enable timeout — exiting.")
        sys.exit(0)

    # ── Read current pose ─────────────────────────────────────────────────────
    def read_current_joints(self):
        """Read joint feedback and initialise self.joints from the real pose."""
        # Allow a moment for feedback messages to arrive after enable
        time.sleep(0.2)
        fb = self.piper.GetArmJointMsgs().joint_state
        self.joints = [
            fb.joint_1 / FACTOR,
            fb.joint_2 / FACTOR,
            fb.joint_3 / FACTOR,
            fb.joint_4 / FACTOR,
            fb.joint_5 / FACTOR,
            fb.joint_6 / FACTOR,
        ]
        print(f"Current joints (deg): {[round(j, 1) for j in self.joints]}")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _go_to_zero(self, timeout: float = 10.0):
        """Move all joints to zero and wait until they are close enough."""
        print("\nReturning to zero position...")
        self.joints = [0.0] * 6
        start = time.time()
        while time.time() - start < timeout:
            self._send_joints()
            fb = self.piper.GetArmJointMsgs().joint_state
            angles = [
                fb.joint_1, fb.joint_2, fb.joint_3,
                fb.joint_4, fb.joint_5, fb.joint_6,
            ]
            if all(abs(a) < 500 for a in angles):   # within 0.5° of zero
                print("At zero position.")
                break
            time.sleep(0.05)
        else:
            print("Zero-position timeout, stopping anyway.")

    def _clamp(self, value: float, idx: int) -> float:        
        lo, hi = JOINT_LIMITS[idx]
        return max(lo, min(hi, value))

    def _send_joints(self):
        # Clamp every joint before sending as a hard safety net
        clamped = [self._clamp(self.joints[i], i) for i in range(6)]
        values = [round(j * FACTOR) for j in clamped]
        self.piper.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
        self.piper.JointCtrl(*values)

    def _toggle_gripper(self):
        """Toggle gripper between open and closed and send the command."""
        self._gripper_open = not self._gripper_open
        angle = self.GRIPPER_OPEN_ANGLE if self._gripper_open else self.GRIPPER_CLOSED_ANGLE
        self.piper.GripperCtrl(angle, self.GRIPPER_EFFORT, 0x01, 0)

    def _print_state(self, label: str = ""):
        j = self.joints
        gripper_str = "OPEN  " if self._gripper_open else "CLOSED"
        sys.stdout.write(
            f"\r{label:14s}  "
            f"J1={j[0]:+7.1f}°  J2={j[1]:+7.1f}°  J3={j[2]:+7.1f}°  "
            f"J4={j[3]:+7.1f}°  J5={j[4]:+7.1f}°  J6={j[5]:+7.1f}°  "
            f"[step={self.step:.0f}°]  [gripper={gripper_str}]   "
        )
        sys.stdout.flush()

    # ── Terminal raw-mode helpers ──────────────────────────────────────────────
    def _set_raw(self):
        self._orig_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

    def _restore_terminal(self):
        if self._orig_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_settings)

    def _read_key(self) -> str:
        """Blocking read with 5 ms timeout. Returns full key sequence or ''."""
        fd = sys.stdin.fileno()
        if not select.select([sys.stdin], [], [], 0.005)[0]:   # 5 ms — loop rate
            return ''
        # Read up to 4 bytes so the full arrow-key sequence arrives in one shot
        return os.read(fd, 4).decode('utf-8', errors='replace')

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("\n=== Piper Keyboard Joint Control ===")
        print(f"  Up / Down  :  Joint 2  ± {self.step:.0f}°")
        print(f"  Left/Right :  Joint 3  ± {self.step:.0f}°")
        print(f"  m  / n     :  Joint 5  ± {self.step:.0f}°")
        print(f"  +  / -     :  Step size  (current: {self.step:.0f}°)")
        print(f"  c          :  Toggle gripper open / closed")
        print(f"  q  / Ctrl+C:  Quit\n")

        self._set_raw()
        try:
            while True:
                key = self._read_key()

                if key == '\x1b[A':           # ↑  joint 2 up
                    self.joints[1] = self._clamp(self.joints[1] + self.step, 1)
                    self._print_state("↑ Joint 2 +")

                elif key == '\x1b[B':         # ↓  joint 2 down
                    self.joints[1] = self._clamp(self.joints[1] - self.step, 1)
                    self._print_state("↓ Joint 2 -")

                elif key == '\x1b[C':         # →  joint 3 up
                    self.joints[2] = self._clamp(self.joints[2] + self.step, 2)
                    self._print_state("→ Joint 3 +")

                elif key == '\x1b[D':         # ←  joint 3 down
                    self.joints[2] = self._clamp(self.joints[2] - self.step, 2)
                    self._print_state("← Joint 3 -")

                elif key in ('m', 'M'):       # m  joint 5 up
                    self.joints[4] = self._clamp(self.joints[4] + self.step, 4)
                    self._print_state("m  Joint 5 +")

                elif key in ('n', 'N'):       # n  joint 5 down
                    self.joints[4] = self._clamp(self.joints[4] - self.step, 4)
                    self._print_state("n  Joint 5 -")

                elif key == '+':              # increase step
                    self.step = min(self.step + 5.0, 45.0)
                    self._print_state(f"step → {self.step:.0f}°")

                elif key == '-':              # decrease step
                    self.step = max(self.step - 5.0, 1.0)
                    self._print_state(f"step → {self.step:.0f}°")

                elif key in ('c', 'C'):        # toggle gripper
                    self._toggle_gripper()
                    label = "c grip OPEN " if self._gripper_open else "c grip CLOSE"
                    self._print_state(label)
                    continue                    # skip _send_joints for gripper-only move

                elif key in ('q', 'Q', '\x03'):  # q or Ctrl+C
                    break

                self._send_joints()
                # no extra sleep — select() in _read_key provides the 5 ms pacing

        finally:
            self._restore_terminal()
            self.piper.GripperCtrl(self.GRIPPER_CLOSED_ANGLE, self.GRIPPER_EFFORT, 0x01, 0)
            self._go_to_zero()
            self.emergency_stop()


def main():
    parser = argparse.ArgumentParser(
        description="Keyboard joint control for the Piper arm"
    )
    parser.add_argument(
        "--can", default="can2",
        help="CAN port to connect to (default: can2)"
    )
    parser.add_argument(
        "--step", type=float, default=5.0,
        help="Initial step size in degrees (default: 5)"
    )
    args = parser.parse_args()

    piper = C_PiperInterface(args.can)
    piper.ConnectPort()

    ctrl = KeyboardJointCtrl(piper, step=args.step)
    ctrl.emergency_restore()
    ctrl.enable()
    ctrl.read_current_joints()
    ctrl.run()


if __name__ == "__main__":
    main()

#sudo modprobe gs_usb
#bash can_activate.sh can2 1000000 1-4.1:1.0
