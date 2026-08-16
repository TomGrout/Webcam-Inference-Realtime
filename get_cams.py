#!/usr/bin/python

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

get_working_cameras()