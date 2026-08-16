# inference_worker.py
import cv2
import time
import queue
import threading
from yolov8 import YOLOv8
import config
from mqtt_handler import publish_detections

def camera_stream_worker(cam_id: int, stream_name: str, backend: int, 
                         display_queue: queue.Queue, stop_event: threading.Event, mqtt_client):
    """
    Runs in a dedicated thread per camera: captures frames, runs YOLOv8 ONNX inference,
    publishes telemetry over MQTT, and passes annotated frames to the main GUI thread.
    """
    cap = cv2.VideoCapture(cam_id, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] [{stream_name}] Failed to open camera index {cam_id}")
        return

    detector = YOLOv8(
        config.MODEL_PATH, 
        conf_thres=config.CONFIDENCE_THRESHOLD, 
        iou_thres=config.IOU_THRESHOLD
    )
    
    prev_time = time.time()
    print(f"[INFO] [{stream_name}] Stream started on index {cam_id}.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference
        boxes, scores, class_ids = detector(frame)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Filter target classes
        detected_objects = []
        for cid, score in zip(class_ids, scores):
            if cid in config.TARGET_CLASSES:
                detected_objects.append({
                    "classId": config.TARGET_CLASSES[cid],
                    "confidence": round(float(score), 2)
                })

        # MQTT Telemetry
        if detected_objects:
            publish_detections(mqtt_client, stream_name, fps, detected_objects, curr_time)

        # Visual Annotations
        annotated_frame = detector.draw_detections(frame)
        cv2.putText(annotated_frame, f"{stream_name} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Thread-safe push to display queue (drops old frame if unread)
        if display_queue.full():
            try:
                display_queue.get_nowait()
            except queue.Empty:
                pass
        display_queue.put(annotated_frame)

    cap.release()
    print(f"[INFO] [{stream_name}] Stream stopped.")