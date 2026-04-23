#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
Keyboard end-effector (IK) control for the Piper arm.

  IK solver : pink + pinocchio
  URDF      : /workspace/piper_ros/src/piper_description/urdf/piper_description.urdf
  Interface : C_PiperInterface_V2

Key bindings
  ↑  / ↓   : Z + / Z−   (mm)
  ← / →    : X − / X+   (mm)
  w  / s   : Y + / Y−   (mm)
  u  / j   : RX + / RX− (deg, body frame)
  i  / k   : RY + / RY− (deg, body frame)
  o  / l   : RZ + / RZ− (deg, body frame)
  c        : Toggle gripper open (70°) / closed (0°)
  + / -    : Increase / decrease step size
  q / Ctrl+C : Quit (triggers emergency stop)
"""

import sys
import os
import tty
import termios
import select
import time
import argparse

import numpy as np
import pinocchio as pin
from pink import Configuration
from pink.tasks import FrameTask
from pink.barriers import PositionBarrier
from pink.limits import ConfigurationLimit, VelocityLimit
import pink

from piper_sdk import C_PiperInterface_V2
from utils import go_to_joints as utils_go_to_joints

# ── Constants ─────────────────────────────────────────────────────────────────
URDF_PATH = "/workspace/piper_ros/src/piper_description/urdf/piper_description.urdf"
EE_FRAME       = "link6"   # IK control frame (wrist joint)
FINGER1_FRAME  = "link7"   # finger 1 tip
FINGER2_FRAME  = "link8"   # finger 2 tip

FACTOR  = 1000   # SDK unit = value × 1000 (degrees for joints, mm for pose)
SPEED   = 20     # Motion speed percentage (0–100)

IK_DT      = 0.02  # IK integration timestep (s)
IK_ITERS   = 8     # Pink solve iterations per keypress
CONTROL_DT = 0.02  # s between interpolation steps sent to the arm

INTERMEDIATE = [0, 34196, -32149, 0, 32955, 0]
HOME         = [0, -0, 0, 0, 20000, 0]

# Arm joint limits (degrees) — for SDK clamping
JOINT_LIMITS = [
    (-154.0, 154.0),   # joint 1
    (   -5.0, 195.0),   # joint 2
    (-175.0,   5.0),   # joint 3
    (-100.0, 112.0),   # joint 4
    ( -75.0,  75.0),   # joint 5
    (-120.0, 120.0),   # joint 6
]


class IKEndEffectorCtrl:
    GRIPPER_OPEN_ANGLE   = 96000   # 70° × 1000
    GRIPPER_CLOSED_ANGLE = 0
    GRIPPER_EFFORT       = 1000    # 1 N·m

    def __init__(
        self,
        piper: C_PiperInterface_V2,
        step_mm: float       = 5.0,
        step_deg: float      = 3.0,
        move_duration: float = 0.5,
        pos_min_m: np.ndarray = None,   # workspace min bounds [x,y,z] in metres
        pos_max_m: np.ndarray = None,   # workspace max bounds [x,y,z] in metres
    ):
        self.piper         = piper
        self.step_mm       = step_mm
        self.step_deg      = step_deg
        self.move_duration = move_duration
        self._gripper_open  = False
        self._orig_settings = None

        # ── Pinocchio / pink setup ──────────────────────────────────────────
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.data  = self.model.createData()
        self.ee_id       = self.model.getFrameId(EE_FRAME)
        self.finger1_id  = self.model.getFrameId(FINGER1_FRAME)
        self.finger2_id  = self.model.getFrameId(FINGER2_FRAME)

        self.ee_task = FrameTask(
            EE_FRAME,
            position_cost=1.0,
            orientation_cost=0.5,
        )

        # ── Workspace position barrier (PositionBarrier on EE_FRAME) ────────
        # Bounds in metres; z has no upper limit so use a large sentinel.
        self._pos_min = pos_min_m if pos_min_m is not None else np.array([ 0.100, -0.200,  0.200])
        self._pos_max = pos_max_m if pos_max_m is not None else np.array([ 0.700,  0.200, 10.000])
        _p_min = self._pos_min
        _p_max = self._pos_max
        self.pos_barrier_f1 = PositionBarrier(
            FINGER1_FRAME,
            indices=[0, 1, 2],
            p_min=_p_min,
            p_max=_p_max,
            gain=10.0,
        )
        self.pos_barrier_f2 = PositionBarrier(
            FINGER2_FRAME,
            indices=[0, 1, 2],
            p_min=_p_min,
            p_max=_p_max,
            gain=10.0,
        )

        # Start with neutral config; will be overwritten after enable
        self.q = pin.neutral(self.model)

        # Target SE3 — initialised after reading current joints
        self.target_pos = None   # np.ndarray (3,) in metres
        self.target_rot = None   # np.ndarray (3,3)

        # Joint indices to hold fixed during IK (e.g. {0} to lock joint1)
        self.locked_joints: set = set()

        # Pre-built limits for the normal (full-model) solve
        self._config_limit = ConfigurationLimit(self.model)
        self._vel_limit    = VelocityLimit(self.model)

        # Reduced model — built on demand when locked_joints is non-empty.
        # Keyed by (frozenset of locked indices, tuple of their rounded values)
        # so it is rebuilt only when the locked set or their positions change.
        self._r_model     = None
        self._r_data      = None
        self._r_ee_task   = None
        self._r_barriers  = None
        self._r_limits    = None
        self._r_free_idxs = None   # list: reduced DOF index → full q index
        self._r_cache_key = None

    # ── Enable ────────────────────────────────────────────────────────────────
    def enable(self, timeout: float = 10.0):
        """Activate the arm using the V2 API."""
        print("Enabling arm...")
        start = time.time()
        while time.time() - start < timeout:
            if self.piper.EnablePiper():
                self.piper.GripperCtrl(0, self.GRIPPER_EFFORT, 0x01, 0)
                print("Arm enabled.")
                return
            time.sleep(0.01)
        print("Enable timeout — exiting.")
        sys.exit(1)

    # ── Emergency restore / stop ──────────────────────────────────────────────
    def emergency_restore(self):
        """Resume from E-stop and put arm into CAN control mode."""
        print("Running emergency restore...")
        self.piper.MotionCtrl_1(0x02, 0, 0x00)   # resume
        self.piper.MotionCtrl_1(0x00, 0, 0x00)   # normal operation
        self.piper.MotionCtrl_2(0x01, 0, 0, 0x00) # standby
        self.piper.GripperCtrl(0, 0, 0x02, 0)
        time.sleep(1)
        self.piper.MotionCtrl_2(0x01, 0, 0, 0x00) # CAN mode
        self.piper.GripperCtrl(0, 0, 0x03, 0)
        time.sleep(1)
        print("Restore done.")

    def emergency_stop(self):
        """Send E-stop command."""
        print("\nSending emergency stop...")
        self.piper.MotionCtrl_1(0x01, 0, 0x00)

    # ── Joint sync ────────────────────────────────────────────────────────────
    def _check_joints_in_workspace(self, values: list) -> bool:
        """
        Run FK on SDK joint values (degrees × 1000) and check the EE position
        against the workspace bounds.  Prints a warning if outside and returns False.
        """
        q_check = pin.neutral(self.model)
        for i in range(6):
            q_check[i] = np.deg2rad(values[i] / FACTOR)
        data_check = self.model.createData()
        pin.forwardKinematics(self.model, data_check, q_check)
        pin.updateFramePlacements(self.model, data_check)
        violations = []
        # Check both finger tips individually (matches the two PositionBarriers)
        for frame_id, frame_name in [
            (self.finger1_id, FINGER1_FRAME),
            (self.finger2_id, FINGER2_FRAME),
        ]:
            pos = data_check.oMf[frame_id].translation
            for i, axis in enumerate(['X', 'Y', 'Z']):
                if pos[i] < self._pos_min[i]:
                    violations.append(
                        f"{frame_name} {axis}={pos[i]*1000:+.1f}mm < min {self._pos_min[i]*1000:+.1f}mm"
                    )
                elif pos[i] > self._pos_max[i]:
                    violations.append(
                        f"{frame_name} {axis}={pos[i]*1000:+.1f}mm > max {self._pos_max[i]*1000:+.1f}mm"
                    )
        if violations:
            print(f"\n[WARNING] go_to_joints target outside workspace: {', '.join(violations)} — proceeding anyway.")
            return False
        return True

    def _sync_pinocchio_from_sdk(self) -> None:
        """Read current joint angles from SDK and update self.q / self.target_pos/rot."""
        fb = self.piper.GetArmJointMsgs().joint_state
        joints_sdk = [
            fb.joint_1, fb.joint_2, fb.joint_3,
            fb.joint_4, fb.joint_5, fb.joint_6,
        ]
        self.q = pin.neutral(self.model)
        for i in range(6):
            self.q[i] = np.deg2rad(joints_sdk[i] / FACTOR)
        self.q[6] = 0.0
        self.q[7] = 0.0
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        T_ee = self.data.oMf[self.ee_id]
        self.target_pos = T_ee.translation.copy()
        self.target_rot = T_ee.rotation.copy()

    def go_to_joints(self, values: list, tolerance: int = 500, timeout: float = 10.0, duration: float = None) -> bool:
        """Move to SDK joint values with smoothstep interpolation, then confirm arrival."""
        if duration is None:
            duration = self.move_duration
        self._check_joints_in_workspace(values)
        fb = self.piper.GetArmJointMsgs().joint_state
        current = [
            fb.joint_1, fb.joint_2, fb.joint_3,
            fb.joint_4, fb.joint_5, fb.joint_6,
        ]
        n_steps = max(1, round(duration / CONTROL_DT))
        for k in range(1, n_steps + 1):
            t = k / n_steps
            t = t * t * (3.0 - 2.0 * t)   # smoothstep
            interp = [round(current[i] + t * (values[i] - current[i])) for i in range(6)]
            self.piper.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
            self.piper.JointCtrl(*interp)
            time.sleep(CONTROL_DT)
        # Wait until the arm settles at the final target
        result = utils_go_to_joints(
            self.piper, values,
            speed=SPEED,
            tolerance=tolerance,
            timeout=timeout,
            verbose=True,
        )
        # Sync pinocchio state so IK starts from the correct configuration
        self._sync_pinocchio_from_sdk()
        return result

    # ── IK solver ─────────────────────────────────────────────────────────────
    def _solve_ik(self) -> list:
        """
        Run pink IK to track self.target_pos / self.target_rot.
        Updates self.q in place and returns arm joint angles in degrees.
        If locked_joints is non-empty, delegates to _solve_ik_reduced() which
        builds a reduced pinocchio model that excludes those joints entirely.
        """
        target_se3 = pin.SE3(self.target_rot, self.target_pos)
        self.ee_task.set_target(target_se3)

        # Clamp self.q to model limits before handing to pink
        for i in range(self.model.nq):
            lo = float(self.model.lowerPositionLimit[i])
            hi = float(self.model.upperPositionLimit[i])
            self.q[i] = float(np.clip(self.q[i], lo, hi))

        if self.locked_joints:
            return self._solve_ik_reduced()

        cfg = Configuration(self.model, self.data, self.q)
        for _ in range(IK_ITERS):
            vel = pink.solve_ik(
                cfg, [self.ee_task], IK_DT,
                solver="daqp",
                barriers=[self.pos_barrier_f1, self.pos_barrier_f2],
                limits=[self._config_limit, self._vel_limit],
            )
            q_next = pin.integrate(self.model, cfg.q, vel * IK_DT)
            for i in range(6):
                lo_rad = np.deg2rad(JOINT_LIMITS[i][0])
                hi_rad = np.deg2rad(JOINT_LIMITS[i][1])
                q_next[i] = float(np.clip(q_next[i], lo_rad, hi_rad))
            q_next[6] = 0.0
            q_next[7] = 0.0
            self.q = q_next
            cfg = Configuration(self.model, self.data, self.q)

        return [np.rad2deg(self.q[i]) for i in range(6)]

    def _solve_ik_reduced(self) -> list:
        """
        IK on a pinocchio reduced model that has locked joints removed.

        pin.buildReducedModel() produces a model with fewer DOF so pink's
        Jacobian, QP matrices, and solution are computed purely over the
        remaining joints — locked joints are invisible to the solver.

        The reduced model is cached and only rebuilt when locked_joints or
        the locked joint positions change.
        """
        cache_key = (
            frozenset(self.locked_joints),
            tuple(round(float(self.q[ji]), 4) for ji in sorted(self.locked_joints)),
        )
        if self._r_cache_key != cache_key:
            # pinocchio joint IDs are 1-indexed; q index ji → joint ID ji+1
            joint_ids  = sorted(ji + 1 for ji in self.locked_joints)
            r_model    = pin.buildReducedModel(self.model, joint_ids, self.q)
            r_data     = r_model.createData()
            r_ee_task  = FrameTask(EE_FRAME, position_cost=1.0, orientation_cost=0.5)
            r_barriers = [
                PositionBarrier(FINGER1_FRAME, indices=[0, 1, 2],
                                p_min=self._pos_min, p_max=self._pos_max, gain=10.0),
                PositionBarrier(FINGER2_FRAME, indices=[0, 1, 2],
                                p_min=self._pos_min, p_max=self._pos_max, gain=10.0),
            ]
            r_limits   = [ConfigurationLimit(r_model), VelocityLimit(r_model)]
            # Map: position in reduced q → position in full q
            free_idxs  = [i for i in range(self.model.nq) if i not in self.locked_joints]
            self._r_model, self._r_data      = r_model, r_data
            self._r_ee_task                  = r_ee_task
            self._r_barriers, self._r_limits = r_barriers, r_limits
            self._r_free_idxs                = free_idxs
            self._r_cache_key                = cache_key

        self._r_ee_task.set_target(pin.SE3(self.target_rot, self.target_pos))

        # Extract the free-joint portion of self.q as the reduced initial config
        r_q = self.q[self._r_free_idxs].copy()
        for i in range(self._r_model.nq):
            lo = float(self._r_model.lowerPositionLimit[i])
            hi = float(self._r_model.upperPositionLimit[i])
            r_q[i] = float(np.clip(r_q[i], lo, hi))

        # Pre-compute index pairs for clamping (only arm joints, not gripper)
        free_arm_pairs = [(r_i, full_i)
                         for r_i, full_i in enumerate(self._r_free_idxs) if full_i < 6]
        grip_r_idxs    = [r_i
                         for r_i, full_i in enumerate(self._r_free_idxs) if full_i >= 6]

        cfg = Configuration(self._r_model, self._r_data, r_q)
        for _ in range(IK_ITERS):
            vel = pink.solve_ik(
                cfg, [self._r_ee_task], IK_DT,
                solver="daqp",
                barriers=self._r_barriers,
                limits=self._r_limits,
            )
            r_q_next = pin.integrate(self._r_model, cfg.q, vel * IK_DT)
            for r_i, full_i in free_arm_pairs:
                lo = np.deg2rad(JOINT_LIMITS[full_i][0])
                hi = np.deg2rad(JOINT_LIMITS[full_i][1])
                r_q_next[r_i] = float(np.clip(r_q_next[r_i], lo, hi))
            for r_i in grip_r_idxs:
                r_q_next[r_i] = 0.0
            r_q = r_q_next
            cfg = Configuration(self._r_model, self._r_data, r_q)

        # Write reduced solution back into full q (locked joints unchanged)
        for r_i, full_i in enumerate(self._r_free_idxs):
            self.q[full_i] = float(r_q[r_i])

        return [np.rad2deg(self.q[i]) for i in range(6)]

    # ── Smooth motion ─────────────────────────────────────────────────────────
    def _smooth_move(self, delta_pos: np.ndarray, delta_rot: np.ndarray, label: str):
        """
        Interpolate from the current target to (target + delta) over
        self.move_duration seconds, sending IK joints every CONTROL_DT seconds.
        Position is linearly interpolated; rotation uses SLERP (log3/exp3).
        Workspace bounds are enforced by self.pos_barrier inside _solve_ik.
        """
        start_pos = self.target_pos.copy()
        start_rot = self.target_rot.copy()
        end_pos   = start_pos + delta_pos
        end_rot   = start_rot @ delta_rot

        omega   = pin.log3(start_rot.T @ end_rot)   # rotation arc vector
        n_steps = max(1, round(self.move_duration / CONTROL_DT))

        for k in range(1, n_steps + 1):
            t = k / n_steps
            t = t * t * (3.0 - 2.0 * t)   # smoothstep: slow→fast→slow
            self.target_pos = start_pos + t * (end_pos - start_pos)
            self.target_rot = start_rot @ pin.exp3(t * omega)
            joints_deg = self._solve_ik()
            self._send_joints(joints_deg)
            self._print_state(label)
            time.sleep(CONTROL_DT)

    # ── Joint send ────────────────────────────────────────────────────────────
    def _send_joints(self, joints_deg: list):
        clamped = [
            max(JOINT_LIMITS[i][0], min(JOINT_LIMITS[i][1], joints_deg[i]))
            for i in range(6)
        ]
        values = [round(d * FACTOR) for d in clamped]
        self.piper.MotionCtrl_2(0x01, 0x01, SPEED, 0x00)
        self.piper.JointCtrl(*values)

    # ── Gripper ───────────────────────────────────────────────────────────────
    def _toggle_gripper(self):
        self._gripper_open = not self._gripper_open
        angle = self.GRIPPER_OPEN_ANGLE if self._gripper_open else self.GRIPPER_CLOSED_ANGLE
        self.piper.GripperCtrl(angle, self.GRIPPER_EFFORT, 0x01, 0)

    # ── TCP helpers ───────────────────────────────────────────────────────────
    def get_tcp_pos(self) -> np.ndarray:
        """Midpoint between the two finger tips in base frame, metres."""
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        return 0.5 * (
            self.data.oMf[self.finger1_id].translation
            + self.data.oMf[self.finger2_id].translation
        )

    def get_tcp_offset_base(self) -> np.ndarray:
        """
        Vector from link6 origin to the gripper midpoint in the base frame (metres).
        Subtract this from a desired midpoint target to get the link6 IK target.
        """
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        midpoint = 0.5 * (
            self.data.oMf[self.finger1_id].translation
            + self.data.oMf[self.finger2_id].translation
        )
        return midpoint - self.data.oMf[self.ee_id].translation

    # ── Display ───────────────────────────────────────────────────────────────
    def _print_state(self, label: str = ""):
        tcp_mm = self.get_tcp_pos() * 1000   # fingertip position in mm
        rpy  = np.rad2deg(pin.rpy.matrixToRpy(self.target_rot))
        grip = "OPEN  " if self._gripper_open else "CLOSED"
        sys.stdout.write(
            f"\r{label:14s}  "
            f"X={tcp_mm[0]:+7.1f}mm  Y={tcp_mm[1]:+7.1f}mm  Z={tcp_mm[2]:+7.1f}mm  "
            f"RX={rpy[0]:+6.1f}°  RY={rpy[1]:+6.1f}°  RZ={rpy[2]:+6.1f}°  "
            f"[t={self.step_mm:.1f}mm  r={self.step_deg:.1f}°]  "
            f"[gripper={grip}]   "
        )
        sys.stdout.flush()

    # ── Raw terminal helpers ──────────────────────────────────────────────────
    def _set_raw(self):
        self._orig_settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

    def _restore_terminal(self):
        if self._orig_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._orig_settings)

    def _read_key(self) -> str:
        """Non-blocking read with 5 ms timeout."""
        if not select.select([sys.stdin], [], [], 0.005)[0]:
            return ''
        return os.read(sys.stdin.fileno(), 4).decode('utf-8', errors='replace')

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        print("\n=== Piper IK End-Effector Control ===")
        print(f"  ↑  / ↓    : Z +  / Z −   (step = {self.step_mm:.1f} mm)")
        print(f"  ← / →     : X −  / X +")
        print(f"  w  / s    : Y +  / Y −")
        print(f"  u  / j    : RX + / RX −  (step = {self.step_deg:.1f}°, body frame)")
        print(f"  i  / k    : RY + / RY −")
        print(f"  o  / l    : RZ + / RZ −")
        print(f"  +  / -    : Step size up / down")
        print(f"  c         : Toggle gripper")
        print(f"  q / Ctrl+C: Quit\n")
        self.go_to_joints(INTERMEDIATE, duration=1.0)
        self._set_raw()
        try:
            while True:
                key = self._read_key()
                if not key:
                    continue

                dm      = self.step_mm  / 1000.0   # mm → m
                dr      = np.deg2rad(self.step_deg)
                label   = ""
                dp      = np.zeros(3)              # position delta (m)
                dR      = np.eye(3)                # rotation delta (body frame)
                do_move = True

                # ── Translation ───────────────────────────────────────────
                if key == '\x1b[A':          # ↑  Z+
                    dp[2] = dm;  label = "↑ Z+"
                elif key == '\x1b[B':        # ↓  Z−
                    dp[2] = -dm; label = "↓ Z−"
                elif key == '\x1b[C':        # →  X+
                    dp[0] = dm;  label = "→ X+"
                elif key == '\x1b[D':        # ←  X−
                    dp[0] = -dm; label = "← X−"
                elif key in ('w', 'W'):      # w  Y+
                    dp[1] = dm;  label = "w Y+"
                elif key in ('s', 'S'):      # s  Y−
                    dp[1] = -dm; label = "s Y−"

                # ── Rotation (body frame) ──────────────────────────────────
                elif key in ('u', 'U'):      # RX+
                    dR = pin.exp3(np.array([ dr, 0.0, 0.0])); label = "u RX+"
                elif key in ('j', 'J'):      # RX−
                    dR = pin.exp3(np.array([-dr, 0.0, 0.0])); label = "j RX−"
                elif key in ('i', 'I'):      # RY+
                    dR = pin.exp3(np.array([0.0,  dr, 0.0])); label = "i RY+"
                elif key in ('k', 'K'):      # RY−
                    dR = pin.exp3(np.array([0.0, -dr, 0.0])); label = "k RY−"
                elif key in ('o', 'O'):      # RZ+
                    dR = pin.exp3(np.array([0.0, 0.0,  dr])); label = "o RZ+"
                elif key in ('l', 'L'):      # RZ−
                    dR = pin.exp3(np.array([0.0, 0.0, -dr])); label = "l RZ−"

                # ── Step size ─────────────────────────────────────────────
                elif key == '+':
                    self.step_mm  = min(self.step_mm  + 2.0, 50.0)
                    self.step_deg = min(self.step_deg + 1.0, 20.0)
                    label = "step ↑";  do_move = False
                elif key == '-':
                    self.step_mm  = max(self.step_mm  - 2.0,  1.0)
                    self.step_deg = max(self.step_deg - 1.0,  1.0)
                    label = "step ↓";  do_move = False

                # ── Gripper ───────────────────────────────────────────────
                elif key in ('c', 'C'):
                    self._toggle_gripper()
                    label = "grip OPEN " if self._gripper_open else "grip CLOSE"
                    self._print_state(label)
                    continue   # gripper only — skip IK/joint send

                # ── Quit ──────────────────────────────────────────────────
                elif key in ('q', 'Q', '\x03'):
                    break

                else:
                    continue

                if do_move:
                    self._smooth_move(dp, dR, label)
                else:
                    self._print_state(label)

        finally:
            self._restore_terminal()
            self.piper.GripperCtrl(self.GRIPPER_CLOSED_ANGLE, self.GRIPPER_EFFORT, 0x01, 0)
            self.go_to_joints(INTERMEDIATE, duration=1.0)
            self.go_to_joints(HOME, duration=1.0)
            self.emergency_stop()


def main():
    parser = argparse.ArgumentParser(
        description="IK end-effector keyboard control for the Piper arm"
    )
    parser.add_argument(
        "--can",      default="can2",
        help="CAN port (default: can2)"
    )
    parser.add_argument(
        "--step-mm",  type=float, default=10.0,
        help="Translation step in mm (default: 5.0)"
    )
    parser.add_argument(
        "--step-deg", type=float, default=3.0,
        help="Rotation step in degrees (default: 3.0)"
    )
    parser.add_argument(
        "--move-duration", type=float, default=0.4,
        help="Seconds to complete one step move (default: 0.4)"
    )
    args = parser.parse_args()

    piper = C_PiperInterface_V2(args.can)
    piper.ConnectPort()
    time.sleep(0.1)

    ctrl = IKEndEffectorCtrl(
        piper,
        step_mm=args.step_mm,
        step_deg=args.step_deg,
        move_duration=args.move_duration,
    )
    ctrl.emergency_restore()
    ctrl.enable()
    ctrl.go_to_joints(HOME, duration=1.0)
    ctrl.run()


if __name__ == "__main__":
    main()
