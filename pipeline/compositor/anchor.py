import cv2
import numpy as np


def track_anchor(video_path: str, seed_point: tuple[float, float]) -> list[tuple[float, float]]:
    """Tracks a single ground-anchor point across the whole video via Lucas-Kanade
    optical flow, starting from a normalized (x, y) seed point on frame 0. Returns a
    normalized (x, y) position per frame — the character's feet are pinned to this
    point each frame, compensating for camera pan/tilt/zoom without any 3D camera
    model. Not real 3D tracking: a poor seed point (e.g. on a moving object rather
    than the ground) will drift."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"Video has no frames: {video_path}")
    h, w = frame.shape[:2]
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    point = np.array([[seed_point[0] * w, seed_point[1] * h]], dtype=np.float32).reshape(-1, 1, 2)
    positions = [(seed_point[0], seed_point[1])]

    lk_params = dict(
        winSize=(31, 31),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    ok, frame = cap.read()
    while ok:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_point, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, point, None, **lk_params)
        if status[0][0] == 1:
            point = new_point
        positions.append((float(point[0, 0, 0]) / w, float(point[0, 0, 1]) / h))
        prev_gray = gray
        ok, frame = cap.read()

    cap.release()
    return positions
