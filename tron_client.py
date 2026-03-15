# tron_client.py
"""
Enhanced TRON1 WebSocket client:
- Persistent WebSocket connection
- Convenience methods for common commands
- Parses/prints `notify_robot_info` telemetry (battery, status, health)

Requires: pip install websocket-client
"""
import json
import os
import time
import uuid
import threading
from typing import Callable, Dict, Any, Optional

try:
    import websocket  # pip install websocket-client
except ImportError as e:
    raise SystemExit("Missing dependency. Run: pip install websocket-client") from e


class TronClient:
    def __init__(
        self,
        ws_url: str = "ws://10.192.1.2:5000",
        accid: str = "WF_TRON1A_377",
        on_status: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        auto_connect: bool = True,
    ) -> None:
        self.ws_url = ws_url
        self.accid = accid
        self.on_status = on_status
        self.on_message_cb = on_message
        self._app = None
        self._thread = None
        self._connected = False
        self._last_status = None
        self._log_twist = os.environ.get("TRON_LOG_TWIST", "0") == "1"
        if auto_connect:
            self.connect()

    @property
    def last_status(self):
        return self._last_status

    def _handle_message(self, _, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            data = {"raw": message}
        title = data.get("title")
        if title == "notify_robot_info":
            d = data.get("data", {})
            status = {
                "battery": d.get("battery"),
                "status": d.get("status"),
                "imu": d.get("imu"),
                "camera": d.get("camera"),
                "motor": d.get("motor"),
                "sw_version": d.get("sw_version"),
                "timestamp": data.get("timestamp"),
            }
            self._last_status = status
            if self.on_status:
                try:
                    self.on_status(status)
                except Exception as e:
                    print("on_status callback error:", e)
        else:
            if self.on_message_cb:
                try:
                    self.on_message_cb(data)
                except Exception as e:
                    print("on_message callback error:", e)
            else:
                print("[TRON RX]", data)

    def _on_open(self, _):
        self._connected = True
        print("[TRON] WebSocket connected:", self.ws_url)

    def _on_close(self, *_):
        self._connected = False
        print("[TRON] WebSocket closed")

    def connect(self) -> None:
        if self._connected:
            return
        self._app = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._handle_message,
            on_close=self._on_close,
        )
        self._thread = threading.Thread(target=self._app.run_forever, daemon=True)
        self._thread.start()
        for _ in range(3):
            if self._connected:
                print("Connected")
                break
            time.sleep(0.1)
        if not self._connected:
            raise ConnectionError(f"Could not connect to {self.ws_url}. Is the robot reachable?")

    @staticmethod
    def _guid() -> str:
        return str(uuid.uuid4())

    def send(self, title: str, data: Optional[Dict[str, Any]] = None) -> None:
        if not self._connected:
            self.connect()
        payload = {
            "accid": self.accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": self._guid(),
            "data": data or {},
        }
        msg = json.dumps(payload, ensure_ascii=False)

        if title != "request_twist" or self._log_twist:
            print(f"[TRON TX] {title} {data or {}}")

        self._app.send(msg)

    # Convenience wrappers
    def stand(self):
        self.send("request_stand_mode")

    def walk(self):
        self.send("request_walk_mode")

    def sit(self):
        self.send("request_sitdown")

    def stair(self):
        self.send("request_stair_mode")

    def emgy_stop(self):
        self.send("request_emgy_stop")

    def enable_imu(self):
        self.send("request_enable_imu")

    def disable_imu(self):
        self.send("request_disable_imu")

    def enable_odometry(self):
        self.send("request_enable_odometry")

    def disable_odometry(self):
        self.send("request_disable_odometry")

    def adjust_height(self, height_m: float):
        self.send("request_adjust_height", {"height": float(height_m)})

    def set_gait(self, gait: str):
        self.send("request_set_gait", {"gait": str(gait)})

    def twist(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.send("request_twist", {"x": float(x), "y": float(y), "z": float(z)})
