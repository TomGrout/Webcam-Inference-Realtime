# tracker.py
import numpy as np

class TrackedObject:
    def __init__(self, track_id: int, box: np.ndarray, class_id: int, score: float):
        self.track_id = track_id
        self.box = box  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.score = score
        self.disappeared_frames = 0
        self.last_centroid = self._get_centroid(box)
        self.has_crossed = False

    def _get_centroid(self, box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def update(self, box: np.ndarray, score: float):
        self.last_centroid = self._get_centroid(self.box)
        self.box = box
        self.score = score
        self.disappeared_frames = 0


class EuclideanIoUTracker:
    """
    Lightweight tracker that associates ONNX YOLO bounding boxes 
    across frames using IoU and centroid distance.
    """
    def __init__(self, max_disappeared=15, iou_threshold=0.3):
        self.next_track_id = 1
        self.tracks = {}  # track_id -> TrackedObject
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def _compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        unionArea = boxAArea + boxBArea - interArea

        return interArea / unionArea if unionArea > 0 else 0.0

    def update(self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
        """
        Updates active tracks with new frame detections.
        Returns: list of active TrackedObject instances.
        """
        if len(boxes) == 0:
            # Increment disappeared counter for all active tracks
            lost_ids = []
            for track_id, track in self.tracks.items():
                track.disappeared_frames += 1
                if track.disappeared_frames > self.max_disappeared:
                    lost_ids.append(track_id)
            for track_id in lost_ids:
                del self.tracks[track_id]
            return list(self.tracks.values())

        if len(self.tracks) == 0:
            # Register all incoming detections as new tracks
            for box, score, cid in zip(boxes, scores, class_ids):
                self.tracks[self.next_track_id] = TrackedObject(self.next_track_id, box, cid, score)
                self.next_track_id += 1
            return list(self.tracks.values())

        # Match existing tracks to detections based on highest IoU
        track_ids = list(self.tracks.keys())
        active_boxes = [self.tracks[tid].box for tid in track_ids]

        iou_matrix = np.zeros((len(track_ids), len(boxes)), dtype=np.float32)
        for i, t_box in enumerate(active_boxes):
            for j, d_box in enumerate(boxes):
                iou_matrix[i, j] = self._compute_iou(t_box, d_box)

        matched_tracks = set()
        matched_detections = set()

        if iou_matrix.size > 0:
            # Greedy matching by maximum IoU
            while True:
                max_val = np.max(iou_matrix)
                if max_val < self.iou_threshold:
                    break
                t_idx, d_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                tid = track_ids[t_idx]
                self.tracks[tid].update(boxes[d_idx], scores[d_idx])
                
                matched_tracks.add(t_idx)
                matched_detections.add(d_idx)
                
                # Invalidate rows and columns
                iou_matrix[t_idx, :] = -1.0
                iou_matrix[:, d_idx] = -1.0

        # Mark unmatched existing tracks as disappeared
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self.tracks[tid].disappeared_frames += 1

        # Purge dead tracks
        dead_tracks = [tid for tid, track in self.tracks.items() if track.disappeared_frames > self.max_disappeared]
        for tid in dead_tracks:
            del self.tracks[tid]

        # Register new detections that didn't match any track
        for d_idx in range(len(boxes)):
            if d_idx not in matched_detections:
                self.tracks[self.next_track_id] = TrackedObject(
                    self.next_track_id, boxes[d_idx], class_ids[d_idx], scores[d_idx]
                )
                self.next_track_id += 1

        return list(self.tracks.values())

class LineCrossingCounter:
    """
    Virtual tripwire counter. Evaluates when an object's centroid crosses 
    a horizontal line (e.g., across a narrow road).
    """
    def __init__(self, line_y: int):
        self.line_y = line_y
        self.inbound_count = 0
        self.outbound_count = 0

    def check_crossing(self, track) -> str | None:
        """
        Returns 'inbound', 'outbound', or None.
        Only registers once per track ID.
        """
        if track.has_crossed:
            return None

        # CURRENT position from the freshly updated bounding box
        curr_x, curr_y = track._get_centroid(track.box)
        
        # PREVIOUS position from memory
        prev_y = track.last_centroid[1]

        # Crossing top-to-bottom
        if prev_y < self.line_y <= curr_y:
            track.has_crossed = True
            self.inbound_count += 1
            return "inbound"

        # Crossing bottom-to-top
        elif prev_y > self.line_y >= curr_y:
            track.has_crossed = True
            self.outbound_count += 1
            return "outbound"

        return None