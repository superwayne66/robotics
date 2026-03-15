"""
Wheel-focused TRON1 manual control.

Key goals:
- Smoother motion via acceleration-limited command ramp.
- Better wheel driving ergonomics (A/D defaults to yaw turn).
- Extendable advanced actions (jump / wall-climb) with safe command mapping.
"""

import argparse
import atexit
import os
import sys
import time
from dataclasses import dataclass

from tron_client import TronClient


if os.name == "nt":
    import msvcrt

    def setup_keyboard() -> None:
        return

    def restore_keyboard() -> None:
        return

    def get_key_nonblocking() -> str | None:
        if not msvcrt.kbhit():
            return None
        ch = msvcrt.getch()
        if ch in (b"\x00", b"\xe0"):
            _ = msvcrt.getch()
            return None
        try:
            return ch.decode("utf-8").lower()
        except Exception:
            return None

else:
    import select
    import termios
    import tty

    _old_term_attrs = None

    def setup_keyboard() -> None:
        global _old_term_attrs
        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY. Run this directly in a terminal.")
        if _old_term_attrs is None:
            fd = sys.stdin.fileno()
            _old_term_attrs = termios.tcgetattr(fd)
            tty.setcbreak(fd)

    def restore_keyboard() -> None:
        global _old_term_attrs
        if _old_term_attrs is not None:
            fd = sys.stdin.fileno()
            termios.tcsetattr(fd, termios.TCSADRAIN, _old_term_attrs)
            _old_term_attrs = None

    def get_key_nonblocking() -> str | None:
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if not dr:
            return None
        ch = sys.stdin.read(1)
        if not ch:
            return None
        return ch.lower()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def approach(current: float, target: float, max_delta: float) -> float:
    if target > current:
        return min(current + max_delta, target)
    return max(current - max_delta, target)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError:
        return default


@dataclass
class Profile:
    name: str
    step_vx: float
    step_vy: float
    step_wz: float
    max_vx: float
    max_vy: float
    max_wz: float
    accel_vx: float
    accel_vy: float
    accel_wz: float


BASE_PROFILES = {
    "precision": Profile("precision", 0.03, 0.03, 0.08, 0.25, 0.20, 0.70, 0.45, 0.45, 0.90),
    "normal": Profile("normal", 0.05, 0.04, 0.12, 0.45, 0.30, 1.10, 0.80, 0.80, 1.60),
    "sport": Profile("sport", 0.08, 0.06, 0.16, 0.70, 0.45, 1.80, 1.40, 1.40, 2.50),
}


def scaled_profile(p: Profile, scale: float) -> Profile:
    s = clamp(scale, 0.05, 1.0)
    return Profile(
        name=p.name,
        step_vx=p.step_vx * s,
        step_vy=p.step_vy * s,
        step_wz=p.step_wz * s,
        max_vx=p.max_vx * s,
        max_vy=p.max_vy * s,
        max_wz=p.max_wz * s,
        accel_vx=p.accel_vx * s,
        accel_vy=p.accel_vy * s,
        accel_wz=p.accel_wz * s,
    )


def profile_from_name(name: str) -> Profile:
    p = BASE_PROFILES[name]
    return Profile(
        name=p.name,
        step_vx=env_float("TRON_STEP_VX", p.step_vx),
        step_vy=env_float("TRON_STEP_VY", p.step_vy),
        step_wz=env_float("TRON_STEP_WZ", p.step_wz),
        max_vx=env_float("TRON_MAX_VX", p.max_vx),
        max_vy=env_float("TRON_MAX_VY", p.max_vy),
        max_wz=env_float("TRON_MAX_WZ", p.max_wz),
        accel_vx=env_float("TRON_ACCEL_VX", p.accel_vx),
        accel_vy=env_float("TRON_ACCEL_VY", p.accel_vy),
        accel_wz=env_float("TRON_ACCEL_WZ", p.accel_wz),
    )


def print_help() -> None:
    print("\nKeymap:")
    print("  1 stand | 2 wheel mode | 3 walk mode | 4 stair mode | 5 sit")
    print("  W/S forward speed target +/- (stair mode defaults to short pulse)")
    print("  A/D yaw rate target +/-   (wheel-friendly turning)")
    print("  Q/E lateral speed target +/-")
    print("  Z/X/C zero vx/vy/wz | T zero all targets")
    print("  P/N/M switch profile: precision/normal/sport")
    print("  G toggle gait trot/pace | R/F height +/-")
    print("  J jump action | K wall-climb action")
    print("  O enable camera | L disable camera")
    print("  V toggle deadman (hold-to-move safety)")
    print("  Y clear motor-fault lock (after hardware check)")
    print("  Space emergency stop | Esc quit | H help")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default=os.environ.get("TRON_WS_URL", "ws://10.192.1.2:5000"))
    ap.add_argument("--accid", default=os.environ.get("TRON_ACCID", "WF_TRON1A_377"))
    ap.add_argument("--profile", choices=["precision", "normal", "sport"], default=os.environ.get("TRON_CTRL_PROFILE", "normal"))
    ap.add_argument("--loop-hz", type=float, default=80.0)
    ap.add_argument("--send-hz", type=float, default=20.0)
    ap.add_argument("--start-mode", choices=["none", "stand", "wheel", "walk", "stair"], default="wheel")
    ap.add_argument("--deadman-timeout", type=float, default=0.25, help="No motion key for this long -> target speed reset")
    ap.add_argument("--stair-scale", type=float, default=0.45, help="Scale profile while in stair mode (0.05~1.0)")
    ap.add_argument(
        "--stair-pulse",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use pulse-based forward/backward command in stair mode",
    )
    ap.add_argument("--stair-step-hold", type=float, default=0.16, help="Pulse duration for stair W/S command")
    ap.add_argument("--stair-step-speed", type=float, default=0.08, help="Pulse speed for stair W/S command")
    ap.add_argument(
        "--require-cam-ok-for-stair",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If camera state is not OK, block stair motion for safety",
    )
    ap.add_argument(
        "--auto-enable-camera",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When stair motion is blocked by camera state, periodically try enable-camera command",
    )
    ap.add_argument(
        "--camera-enable-interval",
        type=float,
        default=2.0,
        help="Seconds between auto camera-enable attempts",
    )
    ap.add_argument(
        "--auto-recover-damping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When status is DAMPING, periodically send stand command to recover",
    )
    ap.add_argument(
        "--damping-recover-interval",
        type=float,
        default=2.0,
        help="Seconds between damping recovery stand attempts",
    )
    args = ap.parse_args()

    profile = profile_from_name(args.profile)
    stair_profile = scaled_profile(profile, args.stair_scale)
    stair_fallback_scale = clamp(args.stair_scale, 0.30, 0.55)
    stair_fallback_profile = scaled_profile(profile, stair_fallback_scale)
    gait = "trot"
    height = 0.50

    tgt_vx = 0.0
    tgt_vy = 0.0
    tgt_wz = 0.0
    cmd_vx = 0.0
    cmd_vy = 0.0
    cmd_wz = 0.0
    locomotion_mode = "wheel"
    deadman_enabled = True
    last_motion_key_time = time.time()
    stair_pulse_until = 0.0
    stair_pulse_dir = 0.0
    last_warn_time = 0.0
    last_camera_enable_try = 0.0
    last_damping_recover_try = 0.0
    camera_cmd_invalid_count = 0
    camera_cmd_unsupported = False
    stair_cmd_unsupported = False
    require_cam_ok_for_stair = args.require_cam_ok_for_stair
    auto_enable_camera = args.auto_enable_camera
    fault_latched = False
    last_fault_warn_time = 0.0

    last_status = {"battery": None, "status": "UNKNOWN", "imu": None, "camera": None, "motor": None}

    def on_status(status):
        last_status.update(status)

    def on_message(msg):
        nonlocal camera_cmd_invalid_count, camera_cmd_unsupported, stair_cmd_unsupported
        nonlocal require_cam_ok_for_stair, auto_enable_camera, locomotion_mode, deadman_enabled
        title = str(msg.get("title", ""))
        data = msg.get("data") or {}
        result = str(data.get("result", "")).lower()
        if not title.startswith("response_"):
            return

        if result == "fail_invalid_cmd":
            if ("camera" in title) or ("rgbd" in title):
                camera_cmd_invalid_count += 1
                if camera_cmd_invalid_count >= 2 and not camera_cmd_unsupported:
                    camera_cmd_unsupported = True
                    auto_enable_camera = False
                    if require_cam_ok_for_stair:
                        require_cam_ok_for_stair = False
                        print(
                            "\n[INFO] Camera enable commands unsupported on this firmware "
                            "(fail_invalid_cmd). Disabled stair camera gate."
                        )
                    else:
                        print("\n[INFO] Camera enable commands unsupported on this firmware.")
            elif any(k in title for k in ("stand", "walk", "stair", "sit")):
                print(f"\n[WARN] Motion command unsupported: {title} result={result}")
                if "stair" in title and not stair_cmd_unsupported:
                    stair_cmd_unsupported = True
                    # Fallback: use WALK runtime mode but keep conservative stair-like limits.
                    used = client.walk()
                    locomotion_mode = "stair_fallback"
                    deadman_enabled = True
                    print(
                        f"\n[FALLBACK] Stair command unsupported. "
                        f"Using WALK runtime ({used}) with stair_fallback controls."
                    )

    setup_keyboard()
    atexit.register(restore_keyboard)

    client = TronClient(ws_url=args.ws_url, accid=args.accid, on_status=on_status, on_message=on_message)

    print("\nWheel control ready")
    print(f"Profile={profile.name} ws={args.ws_url} accid={args.accid}")
    print_help()

    if args.start_mode == "stand":
        used = client.stand()
        locomotion_mode = "stand"
        print(f"[start] stand ({used})")
        time.sleep(0.7)
    elif args.start_mode == "wheel":
        client.wheel_mode()
        locomotion_mode = "wheel"
        print("[start] wheel mode")
        time.sleep(0.7)
    elif args.start_mode == "walk":
        used = client.walk()
        locomotion_mode = "walk"
        print(f"[start] walk mode ({used})")
        time.sleep(0.7)
    elif args.start_mode == "stair":
        used = client.stair()
        locomotion_mode = "stair"
        print(f"[start] stair mode ({used})")
        time.sleep(0.7)
    else:
        locomotion_mode = "none"

    send_period = 1.0 / max(args.send_hz, 1.0)
    loop_period = 1.0 / max(args.loop_hz, 1.0)
    last_send = 0.0
    prev_t = time.time()
    status_period = 0.2
    last_status_print = 0.0

    try:
        while True:
            now = time.time()
            dt = max(1e-4, now - prev_t)
            prev_t = now

            key = get_key_nonblocking()
            if key:
                if key == "\x1b":
                    break
                if key == " ":
                    client.emgy_stop()
                    tgt_vx = tgt_vy = tgt_wz = 0.0
                    cmd_vx = cmd_vy = cmd_wz = 0.0
                    print("\nE-STOP")
                elif key == "1":
                    used = client.stand()
                    locomotion_mode = "stand"
                    print(f"\nMODE -> STAND ({used})")
                elif key == "2":
                    used = client.wheel_mode()
                    locomotion_mode = "wheel"
                    print(f"\nMODE -> WHEEL ({used})")
                elif key == "3":
                    used = client.walk()
                    locomotion_mode = "walk"
                    print(f"\nMODE -> WALK ({used})")
                elif key == "4":
                    if stair_cmd_unsupported:
                        used = client.walk()
                        locomotion_mode = "stair_fallback"
                        deadman_enabled = True
                        print(f"\nMODE -> STAIR_FALLBACK ({used})")
                    else:
                        used = client.stair()
                        locomotion_mode = "stair"
                        deadman_enabled = True
                        print(f"\nMODE -> STAIR ({used})")
                elif key == "5":
                    used = client.sit()
                    locomotion_mode = "sit"
                    print(f"\nMODE -> SIT ({used})")
                elif key == "w":
                    p = stair_profile if locomotion_mode == "stair" else (
                        stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                    )
                    if locomotion_mode in ("stair", "stair_fallback") and args.stair_pulse:
                        stair_pulse_until = now + max(0.02, args.stair_step_hold)
                        stair_pulse_dir = 1.0
                    else:
                        tgt_vx += p.step_vx
                    last_motion_key_time = now
                elif key == "s":
                    p = stair_profile if locomotion_mode == "stair" else (
                        stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                    )
                    if locomotion_mode in ("stair", "stair_fallback") and args.stair_pulse:
                        stair_pulse_until = now + max(0.02, args.stair_step_hold)
                        stair_pulse_dir = -1.0
                    else:
                        tgt_vx -= p.step_vx
                    last_motion_key_time = now
                elif key == "a":
                    p = stair_profile if locomotion_mode == "stair" else (
                        stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                    )
                    tgt_wz += p.step_wz
                    last_motion_key_time = now
                elif key == "d":
                    p = stair_profile if locomotion_mode == "stair" else (
                        stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                    )
                    tgt_wz -= p.step_wz
                    last_motion_key_time = now
                elif key == "q":
                    p = stair_profile if locomotion_mode == "stair" else (
                        stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                    )
                    tgt_vy += p.step_vy
                    last_motion_key_time = now
                elif key == "e":
                    p = stair_profile if locomotion_mode == "stair" else (
                        stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                    )
                    tgt_vy -= p.step_vy
                    last_motion_key_time = now
                elif key == "z":
                    tgt_vx = 0.0
                elif key == "x":
                    tgt_vy = 0.0
                elif key == "c":
                    tgt_wz = 0.0
                elif key == "t":
                    tgt_vx = tgt_vy = tgt_wz = 0.0
                elif key == "g":
                    gait = "pace" if gait == "trot" else "trot"
                    client.set_gait(gait)
                    print(f"\nGAIT -> {gait}")
                elif key == "r":
                    height = min(0.70, height + 0.02)
                    client.adjust_height(height)
                    print(f"\nHEIGHT -> {height:.2f}")
                elif key == "f":
                    height = max(0.35, height - 0.02)
                    client.adjust_height(height)
                    print(f"\nHEIGHT -> {height:.2f}")
                elif key == "j":
                    used = client.jump()
                    print(f"\nACTION -> JUMP ({used})")
                elif key == "k":
                    used = client.wall_climb()
                    print(f"\nACTION -> WALL_CLIMB ({used})")
                elif key == "v":
                    if locomotion_mode in ("stair", "stair_fallback"):
                        deadman_enabled = True
                        print("\nDEADMAN -> ON (forced in stair assist mode)")
                    else:
                        deadman_enabled = not deadman_enabled
                        print(f"\nDEADMAN -> {'ON' if deadman_enabled else 'OFF'}")
                elif key == "y":
                    fault_latched = False
                    print("\nFAULT LOCK -> CLEARED")
                elif key == "o":
                    used = client.enable_camera()
                    print(f"\nCAMERA -> ENABLE ({used})")
                elif key == "l":
                    used = client.disable_camera()
                    print(f"\nCAMERA -> DISABLE ({used})")
                elif key == "p":
                    profile = profile_from_name("precision")
                    stair_profile = scaled_profile(profile, args.stair_scale)
                    stair_fallback_profile = scaled_profile(profile, stair_fallback_scale)
                    print("\nPROFILE -> precision")
                elif key == "n":
                    profile = profile_from_name("normal")
                    stair_profile = scaled_profile(profile, args.stair_scale)
                    stair_fallback_profile = scaled_profile(profile, stair_fallback_scale)
                    print("\nPROFILE -> normal")
                elif key == "m":
                    profile = profile_from_name("sport")
                    stair_profile = scaled_profile(profile, args.stair_scale)
                    stair_fallback_profile = scaled_profile(profile, stair_fallback_scale)
                    print("\nPROFILE -> sport")
                elif key == "h":
                    print_help()

            stair_like = locomotion_mode in ("stair", "stair_fallback")
            active_profile = (
                stair_profile if locomotion_mode == "stair" else (
                    stair_fallback_profile if locomotion_mode == "stair_fallback" else profile
                )
            )
            camera_state = str(last_status.get("camera") or "").upper()
            status_state = str(last_status.get("status") or "").upper()
            motor_state = str(last_status.get("motor") or "")

            if "ERROR" in motor_state.upper():
                if not fault_latched:
                    fault_latched = True
                    client.emgy_stop()
                    print(f"\n[FAULT] Motor error detected: {motor_state}. Motion locked. Press Y after hardware check.")

            # If robot is in damping, movement commands should be zeroed.
            if "DAMPING" in status_state:
                tgt_vx = 0.0
                tgt_vy = 0.0
                tgt_wz = 0.0
                stair_pulse_until = 0.0
                if (
                    args.auto_recover_damping
                    and locomotion_mode in ("stair", "stair_fallback", "walk", "wheel")
                    and (now - last_damping_recover_try >= max(0.5, args.damping_recover_interval))
                ):
                    used = client.stand()
                    print(f"\n[RECOVER] DAMPING -> try STAND ({used})")
                    last_damping_recover_try = now

            if deadman_enabled and (now - last_motion_key_time > args.deadman_timeout):
                tgt_vx = 0.0
                tgt_vy = 0.0
                tgt_wz = 0.0

            if locomotion_mode in ("sit", "stand", "none"):
                tgt_vx = 0.0
                tgt_vy = 0.0
                tgt_wz = 0.0
                stair_pulse_until = 0.0

            # Stair safety gate: require camera OK unless explicitly disabled.
            if stair_like and require_cam_ok_for_stair and camera_state != "OK":
                tgt_vx = 0.0
                tgt_vy = 0.0
                tgt_wz = 0.0
                stair_pulse_until = 0.0
                if now - last_warn_time > 1.5:
                    print("\n[SAFE] CAM != OK, stair motion blocked. Check camera or switch to wheel mode.")
                    last_warn_time = now
                if auto_enable_camera and (now - last_camera_enable_try >= max(0.5, args.camera_enable_interval)):
                    used = client.enable_camera()
                    print(f"\n[AUTO] try enable camera ({used})")
                    last_camera_enable_try = now

            # Stair pulse mode: each W/S press triggers a short, bounded motion pulse.
            if stair_like and args.stair_pulse and now < stair_pulse_until:
                pulse_speed = clamp(abs(args.stair_step_speed), 0.0, active_profile.max_vx)
                tgt_vx = stair_pulse_dir * pulse_speed
                tgt_vy = 0.0
                tgt_wz = 0.0
            elif stair_like and args.stair_pulse and now >= stair_pulse_until:
                tgt_vx = 0.0

            tgt_vx = clamp(tgt_vx, -active_profile.max_vx, active_profile.max_vx)
            tgt_vy = clamp(tgt_vy, -active_profile.max_vy, active_profile.max_vy)
            tgt_wz = clamp(tgt_wz, -active_profile.max_wz, active_profile.max_wz)

            if fault_latched:
                tgt_vx = tgt_vy = tgt_wz = 0.0
                cmd_vx = cmd_vy = cmd_wz = 0.0
                if now - last_fault_warn_time > 1.5:
                    print("\n[SAFE] Fault lock active. Motion blocked. Press Y after fixing motor issue.")
                    last_fault_warn_time = now

            cmd_vx = approach(cmd_vx, tgt_vx, active_profile.accel_vx * dt)
            cmd_vy = approach(cmd_vy, tgt_vy, active_profile.accel_vy * dt)
            cmd_wz = approach(cmd_wz, tgt_wz, active_profile.accel_wz * dt)

            if now - last_send >= send_period:
                client.twist(cmd_vx, cmd_vy, cmd_wz)
                last_send = now

            if now - last_status_print >= status_period:
                sys.stdout.write(
                    "\r"
                    f"BAT:{last_status.get('battery')} MODE:{last_status.get('status')} "
                    f"IMU:{last_status.get('imu')} CAM:{last_status.get('camera')} MOT:{last_status.get('motor')} "
                    f"| profile:{profile.name} mode:{locomotion_mode} deadman:{'on' if deadman_enabled else 'off'} "
                    f"fault:{'locked' if fault_latched else 'clear'} "
                    f"cam_gate:{'on' if require_cam_ok_for_stair else 'off'} "
                    f"cam_cmd:{'unsupported' if camera_cmd_unsupported else 'unknown'} "
                    f"stair_cmd:{'unsupported' if stair_cmd_unsupported else 'unknown'} "
                    f"| tgt({tgt_vx:+.2f},{tgt_vy:+.2f},{tgt_wz:+.2f}) "
                    f"cmd({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f})    "
                )
                sys.stdout.flush()
                last_status_print = now

            sleep_for = loop_period - (time.time() - now)
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        try:
            client.twist(0.0, 0.0, 0.0)
        except Exception:
            pass
        restore_keyboard()
        print("\nExited. Twist reset to zero.")


if __name__ == "__main__":
    main()
