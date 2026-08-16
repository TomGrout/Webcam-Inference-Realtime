# inference_worker.py
import cv2
import time
import queue
import threading
from yolov8 import YOLOv8
import config
from mqtt_handler import publish_detections
from tracker import EuclideanIoUTracker, LineCrossingCounter
import numpy as np

def camera_stream_worker(cam_id: int, stream_name: str, backend: int, 
                         display_queue: queue.Queue, stop_event: threading.Event, mqtt_client):
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
    
    # Instantiate tracker and tripwire for this camera feed
    tracker = EuclideanIoUTracker(max_disappeared=10, iou_threshold=0.3)
    tripwire_y = config.FRAME_HEIGHT // 2  # Horizontal line across middle of feed
    line_counter = LineCrossingCounter(line_y=tripwire_y)

    prev_time = time.time()
    print(f"[INFO] [{stream_name}] Stream started on index {cam_id}.")

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Raw ONNX Inference
        boxes, scores, class_ids = detector(frame)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # 2. Filter Built Environment Target Classes
        filtered_boxes = []
        filtered_scores = []
        filtered_cids = []

        if len(boxes) > 0:
            for b, s, c in zip(boxes, scores, class_ids):
                if c in config.TARGET_CLASSES:
                    filtered_boxes.append(b)
                    filtered_scores.append(s)
                    filtered_cids.append(c)

        # 3. Update Object Tracker
        active_tracks = tracker.update(
            np.array(filtered_boxes), np.array(filtered_scores), np.array(filtered_cids)
        ) if len(filtered_boxes) > 0 else tracker.update(np.empty((0, 4)), np.empty(0), np.empty(0))

        # 4. Check Virtual Line Crossings (Debounced Counts)
        crossed_events = []
        for track in active_tracks:
            direction = line_counter.check_crossing(track)
            if direction:
                class_label = config.TARGET_CLASSES.get(track.class_id, "unknown")
                crossed_events.append({
                    "track_id": track.track_id,
                    "class": class_label,
                    "direction": direction,
                    "confidence": round(float(track.score), 2)
                })

        # 5. Publish to MQTT only when a true line-crossing event occurs
        if crossed_events:
            publish_detections(
                mqtt_client, 
                stream_name, 
                fps, 
                crossed_events, 
                curr_time,
                totals={"inbound": line_counter.inbound_count, "outbound": line_counter.outbound_count}
            )

        # 6. Annotate Visuals (Bounding boxes, IDs, and Tripwire line)
        annotated_frame = frame.copy()
        # Draw tripwire line
        cv2.line(annotated_frame, (0, tripwire_y), (config.FRAME_WIDTH, tripwire_y), (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Tripwire (In: {line_counter.inbound_count} | Out: {line_counter.outbound_count})", 
                    (10, tripwire_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw active tracked objects with their persistent IDs
        for track in active_tracks:
            x1, y1, x2, y2 = track.box.astype(int)
            class_label = config.TARGET_CLASSES.get(track.class_id, "object")
            label_text = f"#{track.track_id} {class_label}"
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, label_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(annotated_frame, f"{stream_name} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Push to main thread queue
        if display_queue.full():
            try:
                display_queue.get_nowait()
            except queue.Empty:
                pass
        display_queue.put(annotated_frame)

    cap.release()
    print(f"[INFO] [{stream_name}] Stream stopped.")