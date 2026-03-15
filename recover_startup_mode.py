"""
Recover robot startup mode from DAMPING and probe usable stand/walk commands.

Usage:
  export TRON_WS_URL=ws://10.192.1.2:5000
  export TRON_ACCID=WF_TRON1A_377
  python recover_startup_mode.py
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any

try:
    import websocket  # type: ignore
except ImportError as exc:
    raise SystemExit("Missing dependency: pip install websocket-client") from exc


def unique(items: list[str]) -> list[str]:
    return list(OrderedDict((x, None) for x in items if x).keys())


def csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default[:]
    return [x.strip() for x in raw.split(",") if x.strip()]


class ProbeClient:
    def __init__(self, ws_url: str, accid: str) -> None:
        self.ws_url = ws_url
        self.accid = accid
        self.ws = None
        self.connected = threading.Event()
        self.msg_lock = threading.Lock()
        self.msgs: list[dict[str, Any]] = []
        self.last_status: dict[str, Any] = {}

    def _on_open(self, _ws):
        self.connected.set()
        print(f"[connected] {self.ws_url}")

    def _on_close(self, *_args):
        print("[closed]")

    def _on_message(self, _ws, message: str):
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        with self.msg_lock:
            self.msgs.append(msg)
        if msg.get("title") == "notify_robot_info":
            data = msg.get("data") or {}
            self.last_status = {
                "battery": data.get("battery"),
                "status": data.get("status"),
                "camera": data.get("camera"),
                "motor": data.get("motor"),
            }

    def connect(self) -> None:
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_close=self._on_close,
        )
        threading.Thread(
            target=lambda: self.ws.run_forever(ping_interval=10, ping_timeout=5),
            daemon=True,
        ).start()
        if not self.connected.wait(4.0):
            raise ConnectionError(f"Cannot connect to {self.ws_url}")

    def close(self) -> None:
        if self.ws:
            self.ws.close()

    def msg_count(self) -> int:
        with self.msg_lock:
            return len(self.msgs)

    def msgs_since(self, idx: int) -> list[dict[str, Any]]:
        with self.msg_lock:
            return list(self.msgs[idx:])

    def send(self, title: str, data: dict[str, Any] | None = None) -> str:
        guid = str(uuid.uuid4())
        payload = {
            "accid": self.accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": guid,
            "data": data or {},
        }
        self.ws.send(json.dumps(payload, ensure_ascii=False))
        print(f"[TX] {title}")
        return guid

    def wait_response(self, req_title: str, guid: str, start_idx: int, timeout: float = 1.8) -> str:
        rsp_title = req_title.replace("request_", "response_", 1)
        end_t = time.time() + timeout
        idx = start_idx
        while time.time() < end_t:
            new_msgs = self.msgs_since(idx)
            if new_msgs:
                idx += len(new_msgs)
                for m in new_msgs:
                    if str(m.get("title", "")) != rsp_title:
                        continue
                    if str(m.get("guid", "")) != guid:
                        continue
                    data = m.get("data") or {}
                    return str(data.get("result", "unknown"))
            time.sleep(0.02)
        return "no_response"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws-url", default=os.environ.get("TRON_WS_URL", "ws://10.192.1.2:5000"))
    ap.add_argument("--accid", default=os.environ.get("TRON_ACCID", "WF_TRON1A_377"))
    args = ap.parse_args()

    stand_candidates = unique(
        csv_env("TRON_CMD_STAND_MODE", [])
        + [
            "request_stand_mode",
            "request_standup",
            "request_recover_stand",
            "request_recover_mode",
            "request_recovery_mode",
        ]
    )
    walk_candidates = unique(
        csv_env("TRON_CMD_WALK_MODE", [])
        + [
            "request_walk_mode",
            "request_move_mode",
            "request_wheel_mode",
        ]
    )

    client = ProbeClient(args.ws_url, args.accid)
    try:
        client.connect()
        time.sleep(1.0)
        print(
            "[status] "
            f"BAT:{client.last_status.get('battery')} "
            f"MODE:{client.last_status.get('status')} "
            f"CAM:{client.last_status.get('camera')} "
            f"MOT:{client.last_status.get('motor')}"
        )

        # Always send E-stop first to keep safe.
        client.send("request_emgy_stop", {})
        time.sleep(0.2)

        print("\n-- Probe STAND commands --")
        stand_ok = None
        for cmd in stand_candidates:
            idx = client.msg_count()
            guid = client.send(cmd, {})
            result = client.wait_response(cmd, guid, idx)
            print(f"{cmd:28s} -> {result}")
            time.sleep(0.25)
            mode = str(client.last_status.get("status", ""))
            if result not in ("fail_invalid_cmd", "fail", "error", "no_response") and "DAMPING" not in mode.upper():
                stand_ok = cmd
                break

        print("\n-- Probe WALK/WHEEL commands --")
        walk_ok = None
        for cmd in walk_candidates:
            idx = client.msg_count()
            guid = client.send(cmd, {})
            result = client.wait_response(cmd, guid, idx)
            print(f"{cmd:28s} -> {result}")
            time.sleep(0.25)
            mode = str(client.last_status.get("status", ""))
            if result not in ("fail_invalid_cmd", "fail", "error", "no_response") and "DAMPING" not in mode.upper():
                walk_ok = cmd
                break

        mode = str(client.last_status.get("status", "UNKNOWN"))
        print("\n=== Recovery Summary ===")
        print(f"final_mode={mode}")
        print(f"stand_ok={stand_ok}")
        print(f"walk_ok={walk_ok}")

        if "DAMPING" in mode.upper():
            print("recover_pass=False")
            print("Robot still in DAMPING. Use official app/remote to recover first, then rerun.")
        else:
            print("recover_pass=True")
            print("Suggested env overrides:")
            if stand_ok:
                print(f"export TRON_CMD_STAND_MODE={stand_ok}")
            if walk_ok:
                print(f"export TRON_CMD_WALK_MODE={walk_ok}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
