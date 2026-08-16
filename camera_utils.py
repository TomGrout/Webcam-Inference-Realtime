# camera_utils.py
import cv2
import platform

def get_os_camera_backend():
    """
    Detects the host OS and selects the optimal OpenCV video backend.
    - Windows: DirectShow (CAP_DSHOW) avoids long timeouts on empty USB indices.
    - Linux / Raspberry Pi: Video4Linux2 (CAP_V4L2) is standard and reliable.
    """
    os_name = platform.system()
    print(f"[INFO] Detected Operating System: {os_name}")

    if os_name == "Windows":
        return cv2.CAP_DSHOW
    elif os_name == "Linux":
        return cv2.CAP_V4L2
    else:
        return cv2.CAP_ANY

def discover_cameras(backend, expected_count=2, max_tests=10):
    """
    Probes USB ports to filter out Linux metadata/dummy nodes and
    returns verified active camera indices.
    """
    print("[INFO] Scanning USB ports for active cameras...")
    working_indices = []

    for i in range(max_tests):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ret, _ = cap.read()  # Test if frame read succeeds
            if ret:
                print(f"  -> Found active camera at index {i}")
                working_indices.append(i)
                if len(working_indices) == expected_count:
                    cap.release()
                    break
        cap.release()

    return working_indices