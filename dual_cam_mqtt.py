#!/usr/bin/python

import cv2
import json
import time
import threading
import queue
import paho.mqtt.client as mqtt
from yolov8 import YOLOv8

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MODEL_PATH = "/home/tom/Projects/Object-Detection/ONNX-YOLOv8-Object-Detection/models/yolov8m.onnx"
#MODEL_PATH = ""

try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
except AttributeError:
    client = mqtt.Client()

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
except Exception as e:
    print(f"[MQTT] Warning: Could not connect to broker ({e}). Continuing inference only.")

# --- 2. DYNAMIC CAMERA DISCOVERY ---
def get_working_cameras(expected_count=2, max_tests=10):
    """Scans indices 0-10 to find active video feeds, ignoring metadata channels."""
    print("Scanning USB ports for active cameras...")
    working_indices = []
    
    for i in range(max_tests):
        # Using CAP_V4L2 forces Linux Video4Linux2 backend for faster probing
        cap = cv2.VideoCapture(i, cv2.CAP_V4L2) 
        if cap.isOpened():
            ret, _ = cap.read() # Actually test if it outputs a frame
            if ret:
                print(f" -> Found active camera at index {i} (/dev/video{i})")
                working_indices.append(i)
                if len(working_indices) == expected_count:
                    cap.release()
                    break
        cap.release()
        
    return working_indices

active_indices = get_working_cameras(expected_count=2)

if len(active_indices) < 2:
    print(f"WARNING: Only found {len(active_indices)} active cameras. Please check USB connections.")

# Assign found indices to your village locations dynamically
LOCATION_NAMES = ["BurroughEnd", "WoodgateHill"]
CAM_CONFIGS = []
for i, idx in enumerate(active_indices):
    if i < len(LOCATION_NAMES):
        CAM_CONFIGS.append({"index": idx, "name": LOCATION_NAMES[i]})
        
        
display_queues = {cfg["name"]: queue.Queue(maxsize=1) for cfg in CAM_CONFIGS}
stop_event = threading.Event()


def camera_stream_worker(cam_id: int, stream_name: str=""):
    cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2) # Added V4L2 backend here too
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
            print(f"[{stream_name}] Failed to open camera index {cam_id}")
            return
    
    detector = YOLOv8(MODEL_PATH, conf_thres=0.6, iou_thres=0.5)
    target_classes = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    prev_time = time.time()
    
    print(f"[{stream_name}] Streaming active on /dev/video{cam_id}")
        
    while not stop_event.is_set():
        ret, frame = cap.read()
        
        if not ret:
            break
        
        boxes, scores, class_ids = detector(frame)
        
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        detected_objects = []
        
        for cid, score in zip(class_ids, scores):
            if cid in target_classes:
                detected_objects.append({
                    "classId": target_classes[cid],
                    "confidence": round(float(score), 2)
                })
        
        if detected_objects:
            payload = {
                "source": stream_name,
                "timestamp": time.time(),
                "fps": round(fps, 1),
                "count": len(detected_objects),
                "objects": detected_objects              
            }
            
            try:
                client.publish(f"dalby/traffic/{stream_name}", json.dumps(payload))
            except Exception:
                pass
            
        combined_img = detector.draw_detections(frame)
        cv2.putText(combined_img, f"{stream_name} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # PUSH TO QUEUE ONLY - NO IMSHOW OR WAITKEY HERE
        q = display_queues[stream_name]
        if q.full():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(combined_img)
        
    cap.release()

    
# Start background workers
threads = []
for cfg in CAM_CONFIGS:
    t = threading.Thread(target=camera_stream_worker, args=(cfg["index"], cfg["name"]), daemon=True)
    t.start()
    threads.append(t)

# Main thread GUI Loop
try:
    while True:
        for cfg in CAM_CONFIGS:
            name = cfg["name"]
            q = display_queues[name]
            if not q.empty():
                frame = q.get()
                cv2.imshow(f"Feed: {name}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    stop_event.set()
    for t in threads:
        t.join(timeout=1.0)
    cv2.destroyAllWindows()
    try:
        client.loop_stop()
        client.disconnect()
    except Exception:
        pass

