import argparse
import time
import numpy as np

import mujoco
import mujoco.viewer
import os
print("RUNNING:", os.path.abspath(__file__))
# ----------------------------
# Tuning notes:
# - Stand pose targets are set by --stand-hip and --stand-knee (see args below).
# - Gait swing targets are computed in the `if args.mode == "gait":` block where qdes["hip_*_Joint"] / qdes["knee_*_Joint"] are assigned.
#   Search for: qdes["hip_L_Joint"], qdes["hip_R_Joint"], qdes["knee_L_Joint"], qdes["knee_R_Joint"].
# Keyboard teleop (TRON1-friendly)
# - WASD / Arrow keys: drive (W/S) + turn (A/D)
# - Space: stop
# - R: reset
# - ESC: quit
#
# macOS NOTE:
#   When using mujoco.viewer.launch_passive on macOS, run via `mjpython` (MuJoCo's python)
#   instead of plain `python`, otherwise the viewer event loop may behave incorrectly.
# ----------------------------

try:
    import glfw  # needed for key codes; viewer uses glfw internally
except Exception as e:
    raise RuntimeError("glfw is required. Please: pip install glfw") from e


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def ctrl_clamp_for_actuator(model: mujoco.MjModel, act_id: int, u: float) -> float:
    """Clamp control value to the actuator ctrlrange if defined."""
    if model.actuator_ctrllimited[act_id]:
        lo, hi = model.actuator_ctrlrange[act_id]
        return clamp(float(u), float(lo), float(hi))
    return float(u)


# --- Helper: Convert desired joint torque to actuator control using gear ---
def ctrl_from_tau(model: mujoco.MjModel, act_id: int, tau: float) -> float:
    """Convert desired joint torque to actuator control.

    Many MJCFs use `motor` actuators with a `gear` scalar. In MuJoCo, the joint torque is
    approximately `gear * ctrl` for a 1-DoF hinge motor. If gear is negative, control
    direction is flipped. We therefore compute `ctrl = tau / gear`.

    Falls back to direct tau if gear is 0/undefined.
    """
    # actuator_gear is (nu, 6). For hinge motors the first component is the scalar gear.
    g = float(model.actuator_gear[act_id, 0]) if model.nu > 0 else 1.0
    if abs(g) < 1e-12:
        u = float(tau)
    else:
        u = float(tau) / g
    return ctrl_clamp_for_actuator(model, act_id, u)


def actuator_indices_by_name(model: mujoco.MjModel, names):
    idx = []
    for n in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if i < 0:
            raise ValueError(f"Actuator name not found: {n}")
        idx.append(i)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to MJCF/XML (or MJB) model file")

    # Timing
    ap.add_argument("--hz", type=float, default=30.0, help="Control update rate (Hz)")
    ap.add_argument(
        "--render-hz",
        type=float,
        default=20.0,
        help="Viewer sync rate (Hz). Lower = less macOS stutter. Try 10-30.",
    )
    ap.add_argument(
        "--max-substeps",
        type=int,
        default=25,
        help="Safety cap on MuJoCo substeps per control tick.",
    )
    ap.add_argument(
        "--substeps",
        type=int,
        default=None,
        help="Override computed substeps per control tick (advanced).",
    )

    # High-level commands (used by gait)
    ap.add_argument("--speed", type=float, default=0.25, help="Drive command magnitude")
    ap.add_argument("--turn", type=float, default=0.20, help="Turn command magnitude")

    # Control mode
    ap.add_argument(
        "--mode",
        choices=["torque", "stand", "gait"],
        default="gait",
        help=(
            "torque: raw actuator commands (usually unstable). "
            "stand: joint PD to hold a pose. "
            "gait: simple alternating-leg gait (demo)."
        ),
    )

    # PD (stand/gait)
    ap.add_argument("--kp", type=float, default=60.0, help="Joint PD Kp (stand/gait)")
    ap.add_argument("--kd", type=float, default=4.0, help="Joint PD Kd (stand/gait)")

    # Standing pose (TRON1 point-foot tends to need more bend)
    ap.add_argument("--stand-hip", type=float, default=0.45, help="Standing hip angle (rad)")
    ap.add_argument("--stand-knee", type=float, default=-0.75, help="Standing knee angle (rad)")

    # Initial base pose (helps prevent immediate collapse)
    ap.add_argument(
        "--init-base-z",
        type=float,
        default=0.65,
        help="Initial base height z (m) used when --init-stand is enabled.",
    )
    ap.add_argument(
        "--init-base-xy",
        type=str,
        default="0,0",
        help="Initial base x,y (m) as 'x,y' used when --init-stand is enabled.",
    )

    # Gait parameters
    ap.add_argument("--stride", type=float, default=0.18, help="Gait amplitude (rad). Start small.")
    ap.add_argument("--gait-hz", type=float, default=0.8, help="Gait frequency (Hz). Start <= 1.0.")

    # Initialization / recovery
    ap.add_argument(
        "--init-stand",
        action="store_true",
        help="Initialize the robot into the standing joint angles before starting (recommended).",
    )
    ap.add_argument(
        "--init-hold-seconds",
        type=float,
        default=0.5,
        help="How long (s) to hold the initial standing pose with PD before accepting keyboard input.",
    )
    ap.add_argument(
        "--recover-on-fall",
        action="store_true",
        help="If the robot falls (base too low), auto-reset and re-initialize into stand pose.",
    )
    ap.add_argument(
        "--fall-z",
        type=float,
        default=0.12,
        help="Fall detection threshold on base height (m). If below, triggers recovery when enabled.",
    )

    # Simple base-orientation balance assist (useful for point-foot bipeds)
    ap.add_argument(
        "--balance",
        action="store_true",
        help="Enable simple balance assist using base orientation (roll/pitch) feedback.",
    )
    ap.add_argument(
        "--bal-kp",
        type=float,
        default=1.8,
        help="Balance proportional gain (maps roll/pitch angle -> joint angle bias).",
    )
    ap.add_argument(
        "--bal-kd",
        type=float,
        default=0.25,
        help="Balance derivative gain (maps base angular velocity -> joint angle bias).",
    )
    ap.add_argument(
        "--bal-max",
        type=float,
        default=0.35,
        help="Max absolute balance joint bias (rad) applied to hip/abad.",
    )
    ap.add_argument(
        "--bal-sign",
        type=float,
        default=1.0,
        help="Sign flip for balance mapping (+1 or -1). Use -1 if robot flips direction.",
    )
    ap.add_argument(
        "--mirror-right",
        choices=["on", "off"],
        default="on",
        help="Whether to mirror right leg joint signs (on=negate right hip/knee, off=same sign as left).",
    )

    # Optional: explicit actuator lists for torque mode
    ap.add_argument(
        "--act-forward",
        default=None,
        help="Comma-separated actuator names/indices used for forward (torque mode).",
    )
    ap.add_argument(
        "--act-yaw",
        default=None,
        help="Comma-separated actuator names/indices used for yaw (torque mode).",
    )

    args = ap.parse_args()
    # Common pitfall: copying commands from rich text can turn "--" into an em-dash "—".
    # If you see "unrecognized arguments: —mode" in the terminal, retype dashes as two hyphens.

    # Load model
    if args.model.endswith(".xml"):
        model = mujoco.MjModel.from_xml_path(args.model)
    else:
        model = mujoco.MjModel.from_binary_path(args.model)
    data = mujoco.MjData(model)

    def set_joint_qpos(joint_name: str, angle: float):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return
        qadr = int(model.jnt_qposadr[jid])
        # Only safe for hinge joints with 1 DoF.
        data.qpos[qadr] = float(angle)

    def base_height() -> float:
        # Prefer a base body if present; otherwise fall back to root free-joint z.
        for bname in ("base", "base_Link", "base_link"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                return float(data.xpos[bid, 2])
        if model.nq >= 3:
            return float(data.qpos[2])
        return 0.0

    def root_free_joint_dofadr() -> int | None:
        free_jids = [i for i in range(model.njnt) if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if not free_jids:
            return None
        return int(model.jnt_dofadr[free_jids[0]])

    def base_rpy() -> tuple[float, float, float]:
        """Return (roll, pitch, yaw) in radians for the root free joint."""
        if model.nq < 7:
            return 0.0, 0.0, 0.0
        qw, qx, qy, qz = (float(data.qpos[3]), float(data.qpos[4]), float(data.qpos[5]), float(data.qpos[6]))
        # Standard quaternion->Euler (roll, pitch, yaw)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = float(np.sign(sinp) * (np.pi / 2.0))
        else:
            pitch = float(np.arcsin(sinp))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        return roll, pitch, yaw

    def base_omega() -> tuple[float, float, float]:
        """Return base angular velocity (wx, wy, wz) for the root free joint."""
        dof0 = root_free_joint_dofadr()
        if dof0 is None or model.nv < dof0 + 6:
            return 0.0, 0.0, 0.0
        # Free joint qvel layout: [vx, vy, vz, wx, wy, wz]
        wx = float(data.qvel[dof0 + 3])
        wy = float(data.qvel[dof0 + 4])
        wz = float(data.qvel[dof0 + 5])
        return wx, wy, wz
    def set_root_free_joint_pose(x: float, y: float, z: float):
        # Find the first free joint and set its (pos, quat) in qpos.
        free_jids = [i for i in range(model.njnt) if int(model.jnt_type[i]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if not free_jids:
            return
        jid = free_jids[0]
        qadr = int(model.jnt_qposadr[jid])
        # qpos layout for free joint: [x, y, z, qw, qx, qy, qz]
        data.qpos[qadr + 0] = float(x)
        data.qpos[qadr + 1] = float(y)
        data.qpos[qadr + 2] = float(z)
        data.qpos[qadr + 3] = 1.0
        data.qpos[qadr + 4] = 0.0
        data.qpos[qadr + 5] = 0.0
        data.qpos[qadr + 6] = 0.0
        # Zero velocities for the free joint DoFs
        dofadr = int(model.jnt_dofadr[jid])
        data.qvel[dofadr : dofadr + 6] = 0.0

    def initialize_to_stand_pose():
        # Put joints into a crouched standing configuration, and place the base at a reasonable height.
        # IMPORTANT: clamp stand angles to joint limits to avoid immediate constraint explosions.
        def clamp_to_joint_range(joint_name: str, angle: float) -> float:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                return float(angle)
            lo, hi = model.jnt_range[jid]
            # Only clamp if joint is limited (MuJoCo stores range regardless)
            if int(model.jnt_limited[jid]) == 1:
                return float(clamp(float(angle), float(lo), float(hi)))
            return float(angle)

        # Parse init base xy
        try:
            x_str, y_str = args.init_base_xy.split(",")
            x0, y0 = float(x_str), float(y_str)
        except Exception:
            x0, y0 = 0.0, 0.0

        # Place the floating base first
        set_root_free_joint_pose(x0, y0, float(args.init_base_z))

        # Standing joint targets (clamped)
        set_joint_qpos("abad_L_Joint", 0.0)
        set_joint_qpos("abad_R_Joint", 0.0)
        set_joint_qpos("hip_L_Joint", clamp_to_joint_range("hip_L_Joint", float(args.stand_hip)))
        # Right leg hip/knee axes are mirrored in this MJCF (see joint axes / ranges), so use opposite sign or not.
        sign = -1.0 if args.mirror_right == "on" else 1.0
        set_joint_qpos("hip_R_Joint", clamp_to_joint_range("hip_R_Joint", sign * float(args.stand_hip)))
        set_joint_qpos("knee_L_Joint", clamp_to_joint_range("knee_L_Joint", float(args.stand_knee)))
        set_joint_qpos("knee_R_Joint", clamp_to_joint_range("knee_R_Joint", sign * float(args.stand_knee)))

        mujoco.mj_forward(model, data)

    if model.nu == 0:
        raise RuntimeError("This model has no actuators (model.nu == 0).")

    # Parse actuator lists (torque mode only)
    def parse_act_list(s):
        if s is None:
            return None
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if not parts:
            return None
        if all(p.isdigit() for p in parts):
            return [int(p) for p in parts]
        return actuator_indices_by_name(model, parts)

    fwd_acts = parse_act_list(args.act_forward) or [0]
    yaw_acts = parse_act_list(args.act_yaw) or ([1] if model.nu >= 2 else [0])

    # Map actuator -> controlled joint qpos/qvel indices (needed for PD)
    def joint_state_indices(joint_name: str):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise ValueError(f"Joint name not found: {joint_name}")
        qpos_adr = int(model.jnt_qposadr[jid])
        dof_adr = int(model.jnt_dofadr[jid])
        return qpos_adr, dof_adr

    act_joint_qpos = [None] * model.nu
    act_joint_qvel = [None] * model.nu
    act_joint_name = [None] * model.nu

    for ai in range(model.nu):
        trnid0 = int(model.actuator_trnid[ai, 0])
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trnid0)
        if jname is None:
            continue
        qadr, vadr = joint_state_indices(jname)
        act_joint_qpos[ai] = qadr
        act_joint_qvel[ai] = vadr
        act_joint_name[ai] = jname

    # Print model info
    print(f"Loaded model: nu={model.nu}, nq={model.nq}, nv={model.nv}")
    print("Actuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        jname = act_joint_name[i]
        g0 = float(model.actuator_gear[i, 0])
        print(f"  [{i}] {name}  (joint={jname}, gear0={g0:+.6g})")
    print("Controls: WASD / Arrow keys, Space=stop, R=reset, ESC=quit")
    print("Tip: start with --mode stand (stable), then switch to --mode gait once standing works.")

    # Timing / substeps
    dt_wall = 1.0 / float(args.hz)
    sim_dt = float(model.opt.timestep)

    auto_substeps = max(1, int(round(dt_wall / max(sim_dt, 1e-9))))
    auto_substeps = min(auto_substeps, int(args.max_substeps))

    substeps = int(args.substeps) if args.substeps is not None else auto_substeps
    substeps = max(1, min(substeps, int(args.max_substeps)))

    render_dt = 1.0 / max(float(args.render_hz), 1e-6)

    print(
        f"[info] model.opt.timestep={sim_dt:.6g}s, control_hz={args.hz}, substeps={substeps}, render_hz={args.render_hz}, mode={args.mode}"
    )

    # Key state
    keys_down = set()

    def key_callback(*cb_args):
        """Normalize MuJoCo viewer key callback signatures."""
        if len(cb_args) == 1:
            key = cb_args[0]
            action = glfw.PRESS
        elif len(cb_args) == 4:
            key, _scancode, action, _mods = cb_args
        elif len(cb_args) == 5:
            _window, key, _scancode, action, _mods = cb_args
        else:
            return

        if action == glfw.REPEAT:
            action = glfw.PRESS

        if action == glfw.PRESS:
            keys_down.add(key)
        elif action == glfw.RELEASE:
            keys_down.discard(key)

    # Gait phase
    phase = 0.0

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # ---- Camera lock (prevents MuJoCo's default WASD camera controls from hijacking teleop) ----
        # If the MJCF defines a camera named "track", lock the viewer to it.
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        if cam_id >= 0:
            try:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = cam_id
                print("[info] Using fixed camera: track")
            except Exception:
                # Older MuJoCo viewer builds may not expose cam controls.
                print("[info] Camera lock not available in this MuJoCo build; use Arrow keys instead of WASD.")
        else:
            print("[info] No camera named 'track' in model; use Arrow keys if WASD moves the camera.")

        # ---- Initialize to a stable standing pose (prevents immediate collapse) ----
        if args.init_stand:
            initialize_to_stand_pose()

            # Hold the pose briefly with PD (no keyboard drive) so it settles.
            hold_until = time.time() + float(args.init_hold_seconds)
            while viewer.is_running() and time.time() < hold_until:
                # PD toward stand pose only
                data.ctrl[:] = 0.0
                kp = float(args.kp)
                kd = float(args.kd)
                qdes_hold = {
                    "hip_L_Joint": args.stand_hip,
                    "knee_L_Joint": args.stand_knee,
                    "hip_R_Joint": (-1.0 if args.mirror_right == "on" else 1.0) * args.stand_hip,
                    "knee_R_Joint": (-1.0 if args.mirror_right == "on" else 1.0) * args.stand_knee,
                    "abad_L_Joint": 0.0,
                    "abad_R_Joint": 0.0,
                }
                for ai in range(model.nu):
                    qadr = act_joint_qpos[ai]
                    vadr = act_joint_qvel[ai]
                    jname = act_joint_name[ai]
                    if qadr is None or vadr is None or jname is None:
                        continue
                    q = float(data.qpos[qadr])
                    qd = float(data.qvel[vadr])
                    q_t = float(qdes_hold.get(jname, 0.0))
                    tau = kp * (q_t - q) + kd * (0.0 - qd)
                    data.ctrl[ai] = ctrl_from_tau(model, ai, tau)

                for _ in range(substeps):
                    mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.001)

        last_ctrl_wall = time.time()
        last_render_wall = last_ctrl_wall

        while viewer.is_running():
            # Small sleeps keep macOS UI responsive
            now = time.time()
            sleep_s = dt_wall - (now - last_ctrl_wall)
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.005))
            else:
                time.sleep(0.001)
            last_ctrl_wall = time.time()

            # Quit/reset
            if glfw.KEY_ESCAPE in keys_down:
                break
            if glfw.KEY_R in keys_down:
                mujoco.mj_resetData(model, data)
                if args.init_stand:
                    initialize_to_stand_pose()
                mujoco.mj_forward(model, data)
                keys_down.discard(glfw.KEY_R)

            # Optional auto-recovery if the robot falls
            if args.recover_on_fall and base_height() < float(args.fall_z):
                print("[warn] Fall detected; resetting + re-initializing stand pose.")
                mujoco.mj_resetData(model, data)
                initialize_to_stand_pose()
                mujoco.mj_forward(model, data)

            # Read drive/turn in [-1, 1]
            drive = 0.0
            turn = 0.0
            if glfw.KEY_W in keys_down or glfw.KEY_UP in keys_down:
                drive += 1.0
            if glfw.KEY_S in keys_down or glfw.KEY_DOWN in keys_down:
                drive -= 1.0
            if glfw.KEY_A in keys_down or glfw.KEY_LEFT in keys_down:
                turn += 1.0
            if glfw.KEY_D in keys_down or glfw.KEY_RIGHT in keys_down:
                turn -= 1.0
            if glfw.KEY_SPACE in keys_down:
                drive = 0.0
                turn = 0.0

            # Control
            data.ctrl[:] = 0.0

            if args.mode == "torque":
                # Raw torque commands (usually not stable for locomotion)
                cmd_fwd = float(args.speed) * drive
                cmd_yaw = float(args.turn) * turn

                for i in fwd_acts:
                    if 0 <= i < model.nu:
                        data.ctrl[i] = ctrl_clamp_for_actuator(model, i, cmd_fwd)
                for i in yaw_acts:
                    if 0 <= i < model.nu:
                        data.ctrl[i] = ctrl_clamp_for_actuator(model, i, cmd_yaw)

            else:
                # Joint-space PD for joints that have actuators
                kp = float(args.kp)
                kd = float(args.kd)

                # Base stand targets (NOTE: right leg is mirrored in this model)
                sign = -1.0 if args.mirror_right == "on" else 1.0
                qdes = {
                    "hip_L_Joint": args.stand_hip,
                    "knee_L_Joint": args.stand_knee,
                    "hip_R_Joint": sign * args.stand_hip,
                    "knee_R_Joint": sign * args.stand_knee,
                    "abad_L_Joint": 0.0,
                    "abad_R_Joint": 0.0,
                }
                qd_des = {k: 0.0 for k in qdes}

                # Optional balance assist: bias hip/abad targets based on base roll/pitch.
                # For point-foot bipeds, pure joint PD to a fixed pose often falls.
                if args.balance:
                    roll, pitch, _yaw = base_rpy()
                    wx, wy, _wz = base_omega()

                    # Map pitch (forward/back lean) to hip flexion; map roll (side lean) to abduction.
                    # Use angular velocity damping for smoother response.
                    kp_b = float(args.bal_kp)
                    kd_b = float(args.bal_kd)
                    max_b = float(args.bal_max)

                    bal_sign = float(args.bal_sign)
                    hip_bias = clamp(bal_sign * (kp_b * pitch + kd_b * wy), -max_b, max_b)
                    abad_bias = clamp(bal_sign * (kp_b * roll + kd_b * wx), -max_b, max_b)

                    # Apply symmetric biases (right side is mirrored in joint coordinates).
                    qdes["hip_L_Joint"] = float(qdes.get("hip_L_Joint", 0.0)) + hip_bias
                    qdes["hip_R_Joint"] = float(qdes.get("hip_R_Joint", 0.0)) - hip_bias

                    qdes["abad_L_Joint"] = float(qdes.get("abad_L_Joint", 0.0)) + abad_bias
                    qdes["abad_R_Joint"] = float(qdes.get("abad_R_Joint", 0.0)) - abad_bias

                if args.mode == "gait":
                    # Simple alternating swing: left/right out of phase
                    gait_hz = float(args.gait_hz) * (1.0 + 0.25 * abs(drive))
                    phase = (phase + 2.0 * np.pi * gait_hz * dt_wall) % (2.0 * np.pi)

                    stride = float(args.stride) * clamp(abs(drive), 0.0, 1.0)

                    # Allow turning in-place (even when drive==0): add small abduction bias.
                    abad_bias = 0.18 * float(args.turn) * clamp(turn, -1.0, 1.0)
                    qdes["abad_L_Joint"] = +abad_bias
                    qdes["abad_R_Joint"] = -abad_bias

                    sL = np.sin(phase)
                    sR = np.sin(phase + np.pi)

                    hip_amp = 0.55 * stride
                    knee_amp = 0.85 * stride

                    # Left leg (as-is)
                    qdes["hip_L_Joint"] = args.stand_hip + hip_amp * sL
                    qdes["knee_L_Joint"] = args.stand_knee - knee_amp * sL

                    # Right leg (mirrored axes -> opposite sign for hip/knee, or not)
                    sign = -1.0 if args.mirror_right == "on" else 1.0
                    qdes["hip_R_Joint"] = sign * (args.stand_hip + hip_amp * sR)
                    qdes["knee_R_Joint"] = sign * (args.stand_knee - knee_amp * sR)

                    omega = 2.0 * np.pi * gait_hz
                    qd_des["hip_L_Joint"] = omega * hip_amp * np.cos(phase)
                    qd_des["knee_L_Joint"] = -omega * knee_amp * np.cos(phase)

                    # Right leg mirrored -> desired velocity also flips sign or not
                    sign = -1.0 if args.mirror_right == "on" else 1.0
                    qd_des["hip_R_Joint"] = sign * (omega * hip_amp * np.cos(phase + np.pi))
                    qd_des["knee_R_Joint"] = sign * (-omega * knee_amp * np.cos(phase + np.pi))

                # Apply PD per actuator
                for ai in range(model.nu):
                    qadr = act_joint_qpos[ai]
                    vadr = act_joint_qvel[ai]
                    jname = act_joint_name[ai]
                    if qadr is None or vadr is None or jname is None:
                        continue

                    q = float(data.qpos[qadr])
                    qd = float(data.qvel[vadr])
                    q_t = float(qdes.get(jname, 0.0))
                    qd_t = float(qd_des.get(jname, 0.0))

                    tau = kp * (q_t - q) + kd * (qd_t - qd)
                    data.ctrl[ai] = ctrl_from_tau(model, ai, tau)

            # Step physics
            for _ in range(substeps):
                mujoco.mj_step(model, data)

                # Auto-recover if simulation explodes
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    print("[warn] NaN/Inf detected; resetting simulation.")
                    mujoco.mj_resetData(model, data)
                    mujoco.mj_forward(model, data)
                    break

            # Render throttle
            now_r = time.time()
            if (now_r - last_render_wall) >= render_dt:
                viewer.sync()
                last_render_wall = now_r

    time.sleep(0.05)
    print("Exited.")




if __name__ == "__main__":
    main()