#!/usr/bin/env python3
# main.py

import cv2
import queue
import threading
import config
from mqtt_handler import init_mqtt
from camera_utils import get_os_camera_backend, discover_cameras
from inference_worker import camera_stream_worker

def main():
    # 1. Initialize Network / MQTT
    mqtt_client = init_mqtt()

    # 2. Hardware Detection (OS-Specific)
    backend = get_os_camera_backend()
    active_indices = discover_cameras(backend, expected_count=config.EXPECTED_CAMS)

    if len(active_indices) < config.EXPECTED_CAMS:
        print(f"[WARN] Found {len(active_indices)} camera(s), expected {config.EXPECTED_CAMS}. Continuing with detected devices.")

    # 3. Setup Streams and Threading Queues
    cam_configs = []
    display_queues = {}
    stop_event = threading.Event()

    for i, idx in enumerate(active_indices):
        if i < len(config.LOCATION_NAMES):
            name = config.LOCATION_NAMES[i]
            cam_configs.append({"index": idx, "name": name})
            display_queues[name] = queue.Queue(maxsize=1)

    # 4. Launch Worker Threads
    threads = []
    for cfg in cam_configs:
        t = threading.Thread(
            target=camera_stream_worker,
            args=(
                cfg["index"],
                cfg["name"],
                backend,
                display_queues[cfg["name"]],
                stop_event,
                mqtt_client
            ),
            daemon=True
        )
        t.start()
        threads.append(t)

    print("\n[INFO] Feeds running. Press 'q' in any window to exit.\n")

    # 5. Main Thread GUI Event Loop (Required for Debian, Linux X11/Wayland & Windows)
    try:
        while True:
            for cfg in cam_configs:
                name = cfg["name"]
                q = display_queues[name]
                if not q.empty():
                    frame = q.get()
                    cv2.imshow(f"Feed: {name}", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Exit key pressed.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received.")
    finally:
        # 6. Graceful Cleanup
        print("[INFO] Cleaning up resources...")
        stop_event.set()

        for t in threads:
            t.join(timeout=2.0)

        cv2.destroyAllWindows()

        if mqtt_client:
            try:
                mqtt_client.loop_stop()
                mqtt_client.disconnect()
            except Exception:
                pass

        print("[INFO] System shutdown complete.")

if __name__ == "__main__":
    main()