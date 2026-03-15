# camera_trigger.py
"""
TRON1 camera-triggered actions with pluggable detectors.

Detectors (choose via MODEL_PROVIDER):
  - "none" (default): simple brightness heuristic
  - "ultralytics": uses Ultralytics YOLO (pip install ultralytics)
  - "onnx": uses ONNX Runtime (pip install onnxruntime) with MODEL_PATH

Camera backends:
  - CAMERA_BACKEND=opencv   (default; OpenCV VideoCapture)
    * CAMERA_DEVICE=0 or RTSP/URL
  - CAMERA_BACKEND=ros
    * CAMERA_TOPIC=/camera/color/image_raw

Action policy:
  - If `hazard == True` -> send EMERGENCY STOP once (latched),
    then periodically zero twist for HOLD_SECONDS.
  - Optional cool-down so we don't spam the bus.

Env vars:
  TRON_WS_URL, TRON_ACCID
  CAMERA_BACKEND=opencv|ros          (default: opencv)
  CAMERA_DEVICE=0 / rtsp://...       (opencv)
  CAMERA_TOPIC=/camera/color/image_raw (ros)

  MODEL_PROVIDER=none|ultralytics|onnx  (default: none)
  MODEL_PATH=path/to/model.onnx         (for onnx)
  MODEL_CLASSES=person,car              (comma list; only for YOLO providers)
  MODEL_SCORE_THR=0.5
  MODEL_NMS_IOU=0.45

  HOLD_SECONDS=1.5          (how long we keep re-sending zero twist after stop)
  COOLDOWN_SECONDS=0.5      (minimum time between consecutive zero-twist sends)
"""
import os, time, sys, math
from typing import List, Tuple, Optional
from tron_client import TronClient

# ------------------------------
# Config helpers
# ------------------------------
def _get_env_float(k: str, default: float) -> float:
    try:
        return float(os.environ.get(k, default))
    except:
        return default

def _get_env_int(k: str, default: int) -> int:
    try:
        return int(os.environ.get(k, default))
    except:
        return default

def _get_env_list(k: str, default: List[str]) -> List[str]:
    v = os.environ.get(k, None)
    if not v:
        return default
    return [x.strip() for x in v.split(",") if x.strip()]

def _get_env_bool(k: str, default: bool) -> bool:
    v = os.environ.get(k, None)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

BACKEND = os.environ.get("CAMERA_BACKEND", "opencv")
MODEL_PROVIDER = os.environ.get("MODEL_PROVIDER", "none").lower()
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MODEL_CLASSES = set([c.lower() for c in _get_env_list("MODEL_CLASSES", ["person"])])
MODEL_SCORE_THR = _get_env_float("MODEL_SCORE_THR", 0.5)
MODEL_NMS_IOU = _get_env_float("MODEL_NMS_IOU", 0.45)
HOLD_SECONDS = _get_env_float("HOLD_SECONDS", 1.5)
COOLDOWN_SECONDS = _get_env_float("COOLDOWN_SECONDS", 0.5)
CAMERA_ENABLE_ON_START = _get_env_bool("CAMERA_ENABLE_ON_START", True)
CAMERA_ENABLE_ONLY = _get_env_bool("CAMERA_ENABLE_ONLY", False)
CAMERA_ENABLE_WAIT = _get_env_float("CAMERA_ENABLE_WAIT", 0.8)
CAMERA_ENABLE_COMMANDS = _get_env_list(
    "CAMERA_ENABLE_COMMANDS",
    [
        "request_enable_camera",
        "request_enable_cam",
        "request_open_camera",
        "request_camera_on",
        "request_start_camera",
        "request_camera_start",
        "request_enable_rgbd",
    ],
)

# ------------------------------
# Detector interface
# ------------------------------
class Detector:
    def infer(self, frame) -> Tuple[bool, List[Tuple[int,int,int,int,str,float]]]:
        """Return (hazard_flag, detections).
        detections: list of (x1,y1,x2,y2,cls_name,score)
        """
        raise NotImplementedError

# Heuristic detector (no deps)
class HeuristicBrightness(Detector):
    def __init__(self, roi=(0.4,0.4,0.6,0.6), thr: float = 210.0):
        self.roi = roi  # (x1_rel,y1_rel,x2_rel,y2_rel)
        self.thr = thr

    def infer(self, frame):
        import cv2
        h, w = frame.shape[:2]
        x1 = int(w * self.roi[0]); y1 = int(h * self.roi[1])
        x2 = int(w * self.roi[2]); y2 = int(h * self.roi[3])
        roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_val = float(gray.mean())
        hazard = mean_val > self.thr
        dets = [ (x1,y1,x2,y2,"bright",min(1.0, mean_val/255.0)) ]
        return hazard, dets

# Ultralytics YOLO (pip install ultralytics)
class UltralyticsYOLO(Detector):
    def __init__(self, classes: set, score_thr: float):
        try:
            from ultralytics import YOLO
        except Exception as e:
            raise RuntimeError("Ultralytics not available. pip install ultralytics") from e
        # By default load yolo11n if no MODEL_PATH is provided
        weights = MODEL_PATH if MODEL_PATH else "yolo11n.pt"
        self.model = YOLO(weights)
        self.classes = set([c.lower() for c in classes])
        self.score_thr = float(score_thr)

    def infer(self, frame):
        import cv2
        results = self.model.predict(source=frame, verbose=False)
        dets = []
        hazard = False
        if not results:
            return False, dets
        res = results[0]
        if res.boxes is None:
            return False, dets
        for b in res.boxes:
            xyxy = b.xyxy[0].tolist()
            x1,y1,x2,y2 = [int(v) for v in xyxy]
            score = float(b.conf[0].item())
            cls_id = int(b.cls[0].item())
            cls_name = res.names.get(cls_id, str(cls_id)).lower()
            if score >= self.score_thr:
                dets.append((x1,y1,x2,y2,cls_name,score))
                if (not self.classes) or (cls_name in self.classes):
                    hazard = True
        return hazard, dets

# ONNX Runtime YOLO (generic; expects standard YOLOv5/8-like output)
class ONNXYOLO(Detector):
    def __init__(self, model_path: str, classes: set, score_thr: float, iou_thr: float):
        if not model_path:
            raise RuntimeError("MODEL_PATH is required for MODEL_PROVIDER=onnx")
        try:
            import onnxruntime as ort
            import numpy as np
        except Exception as e:
            raise RuntimeError("ONNX Runtime not available. pip install onnxruntime") from e
        self.np = __import__("numpy")
        self.ort = ort
        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.i_h, self.i_w = 640, 640  # common default; adjust if your model differs
        self.classes = set([c.lower() for c in classes])
        self.score_thr = float(score_thr)
        self.iou_thr = float(iou_thr)

    def letterbox(self, img, new_shape=(640,640)):
        import cv2, numpy as np
        h, w = img.shape[:2]
        r = min(new_shape[0] / h, new_shape[1] / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        top = (new_shape[0] - nh) // 2
        left = (new_shape[1] - nw) // 2
        out = self.np.full((new_shape[0], new_shape[1], 3), 114, dtype=self.np.uint8)
        resized = cv2.resize(img, (nw, nh))
        out[top:top+nh, left:left+nw] = resized
        return out, r, left, top

    def nms(self, boxes, scores, iou_thr):
        # simple Python NMS (for small batches)
        keep = []
        idxs = self.np.argsort(-scores)
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            ious = self.iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious <= iou_thr]
        return keep

    def iou(self, box, boxes):
        # box: [x1,y1,x2,y2], boxes: Nx4
        x1 = self.np.maximum(box[0], boxes[:,0])
        y1 = self.np.maximum(box[1], boxes[:,1])
        x2 = self.np.minimum(box[2], boxes[:,2])
        y2 = self.np.minimum(box[3], boxes[:,3])
        inter = self.np.maximum(0, x2 - x1) * self.np.maximum(0, y2 - y1)
        area1 = (box[2]-box[0]) * (box[3]-box[1])
        area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        union = area1 + area2 - inter + 1e-6
        return inter / union

    def infer(self, frame):
        import cv2, numpy as np
        img, r, left, top = self.letterbox(frame, (self.i_h,self.i_w))
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2,0,1))[None, ...]  # 1x3xHxW
        inputs = { self.sess.get_inputs()[0].name: x }
        out = self.sess.run(None, inputs)
        # Expect shape [1, num, 6] -> x1,y1,x2,y2,score,class
        pred = out[0][0]
        boxes = []; scores = []; classes = []
        for p in pred:
            x1,y1,x2,y2,score,cls = p.tolist()
            if score < self.score_thr: 
                continue
            # map back to original frame
            x1 = (x1 - left) / r; y1 = (y1 - top) / r
            x2 = (x2 - left) / r; y2 = (y2 - top) / r
            boxes.append([x1,y1,x2,y2]); scores.append(score); classes.append(int(cls))
        if not boxes:
            return False, []

        boxes = np.array(boxes); scores = np.array(scores); classes = np.array(classes)
        keep = self.nms(boxes, scores, self.iou_thr)
        boxes = boxes[keep]; scores = scores[keep]; classes = classes[keep]

        # Map class ids to names if you have a list; otherwise use ids
        dets = []
        hazard = False
        for (x1,y1,x2,y2), sc, cid in zip(boxes, scores, classes):
            cls_name = str(int(cid))
            dets.append((int(x1),int(y1),int(x2),int(y2),cls_name,float(sc)))
            if (not self.classes) or (cls_name in self.classes):
                hazard = True
        return hazard, dets

def build_detector() -> Detector:
    if MODEL_PROVIDER == "ultralytics":
        return UltralyticsYOLO(MODEL_CLASSES, MODEL_SCORE_THR)
    elif MODEL_PROVIDER == "onnx":
        return ONNXYOLO(MODEL_PATH, MODEL_CLASSES, MODEL_SCORE_THR, MODEL_NMS_IOU)
    else:
        return HeuristicBrightness()


def try_enable_camera(client: TronClient):
    """Best-effort camera enable routine for firmware variants."""
    if not CAMERA_ENABLE_ON_START:
        return
    print("[camera] trying enable commands:", ", ".join(CAMERA_ENABLE_COMMANDS))
    for cmd in CAMERA_ENABLE_COMMANDS:
        try:
            client.send(cmd)
        except Exception as e:
            print(f"[camera] send failed for {cmd}: {e}")
            continue

        time.sleep(max(0.05, CAMERA_ENABLE_WAIT))
        st = client.last_status or {}
        cam = str(st.get("camera", "UNKNOWN")).upper()
        mode = st.get("status")
        print(f"[camera] after {cmd}: camera={cam} mode={mode}")
        if cam == "OK":
            print(f"[camera] camera enabled by {cmd}")
            return

    print("[camera] camera still not OK after all commands.")
    print("[camera] If needed, override CAMERA_ENABLE_COMMANDS with your firmware command.")

# ------------------------------
# Drawing utils
# ------------------------------
def draw_dets(frame, dets):
    import cv2
    for (x1,y1,x2,y2,cls_name,score) in dets:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{cls_name}:{score:.2f}", (x1, max(0,y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

# ------------------------------
# OpenCV backend
# ------------------------------
def run_with_opencv(client: TronClient):
    import cv2
    det = build_detector()
    src = os.environ.get("CAMERA_DEVICE", "0")
    try: src = int(src)
    except: pass
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to open camera:", src); return

    print(f"[opencv] MODEL_PROVIDER={MODEL_PROVIDER} classes={list(MODEL_CLASSES)} thr={MODEL_SCORE_THR}")
    stop_latch = False
    hold_until = 0.0
    last_zero_send = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02); continue

        hazard, dets = det.infer(frame)

        # Latch stop if hazard
        now = time.time()
        if hazard and not stop_latch:
            print("Hazard detected -> EMERGENCY STOP")
            client.emgy_stop()
            stop_latch = True
            hold_until = now + HOLD_SECONDS
            last_zero_send = 0.0  # allow immediate zero twist

        # While latched, keep zeroing twist a few times
        if stop_latch and now < hold_until and now - last_zero_send > COOLDOWN_SECONDS:
            client.twist(0.0, 0.0, 0.0)
            last_zero_send = now

        # (Optional) unlatch when hold passes; keep simple for safety
        if stop_latch and now >= hold_until:
            # remain stopped; operator resumes manually
            pass

        draw_dets(frame, dets)
        cv2.putText(frame, f"hazard={hazard} latched={stop_latch}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("camera_trigger", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc
            break

    cap.release()
    try: cv2.destroyAllWindows()
    except: pass

# ------------------------------
# ROS backend
# ------------------------------
def run_with_ros(client: TronClient):
    try:
        import rospy
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        import cv2
    except Exception as e:
        print("ROS backend requires rospy, sensor_msgs, cv_bridge, OpenCV:", e); return

    det = build_detector()
    topic = os.environ.get("CAMERA_TOPIC", "/camera/color/image_raw")
    bridge = CvBridge()
    stop_latch = {"v": False}
    hold_until = {"t": 0.0}
    last_zero_send = {"t": 0.0}

    def cb(msg):
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hazard, dets = det.infer(frame)

        now = time.time()
        if hazard and not stop_latch["v"]:
            print("Hazard detected -> EMERGENCY STOP")
            client.emgy_stop()
            stop_latch["v"] = True
            hold_until["t"] = now + HOLD_SECONDS
            last_zero_send["t"] = 0.0

        if stop_latch["v"] and now < hold_until["t"] and now - last_zero_send["t"] > COOLDOWN_SECONDS:
            client.twist(0.0, 0.0, 0.0)
            last_zero_send["t"] = now

        # Debug view if running in a desktop ROS
        try:
            draw_dets(frame, dets)
            cv2.imshow("camera_trigger_ros", frame)
            cv2.waitKey(1)
        except:
            pass

    rospy.init_node("tron_camera_trigger", anonymous=True)
    rospy.Subscriber(topic, Image, cb, queue_size=1)
    print(f"[ros] subscribing {topic}; MODEL_PROVIDER={MODEL_PROVIDER}")
    rospy.spin()

# ------------------------------
# Entry
# ------------------------------
def main():
    ws_url = os.environ.get("TRON_WS_URL", "ws://10.192.1.2:5000")
    accid = os.environ.get("TRON_ACCID", "PF_TRON1A_075")
    client = TronClient(ws_url=ws_url, accid=accid)
    try_enable_camera(client)
    if CAMERA_ENABLE_ONLY:
        print("[camera] CAMERA_ENABLE_ONLY=1, exiting after enable attempts.")
        return
    if BACKEND.lower() == "ros":
        run_with_ros(client)
    else:
        run_with_opencv(client)

if __name__ == "__main__":
    main()
