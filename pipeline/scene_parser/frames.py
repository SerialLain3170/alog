import cv2
import numpy as np

from pipeline.schemas import BBox, CameraMotion


def open_video(video_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def video_meta(video_path: str) -> tuple[float, float, int]:
    """Returns (fps, duration_s, frame_count)."""
    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frame_count / fps if fps else 0.0
    return fps, duration, frame_count


def sample_keyframes(video_path: str, interval_s: float) -> list[tuple[float, np.ndarray]]:
    """Uniformly sampled (timestamp, BGR frame) pairs at `interval_s` spacing."""
    fps, duration, _ = video_meta(video_path)
    cap = open_video(video_path)
    frames = []
    t = 0.0
    while t < duration:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        frames.append((t, frame))
        t += interval_s
    cap.release()
    return frames


def compute_camera_motion(video_path: str, sample_stride: int = 5) -> list[CameraMotion]:
    """Coarse global optical-flow signal (pan direction + magnitude) sampled every
    `sample_stride` frames. Dense per-frame flow isn't needed for this purpose and
    would be far slower than the rest of the pipeline."""
    fps, _, _ = video_meta(video_path)
    cap = open_video(video_path)

    motions: list[CameraMotion] = []
    prev_gray = None
    diag = None
    idx = 0
    ok, frame = cap.read()
    while ok:
        if idx % sample_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if diag is None:
                diag = float(np.hypot(*gray.shape[:2]))
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 2, 15, 3, 5, 1.2, 0
                )
                dx = float(np.median(flow[..., 0])) / gray.shape[1]
                dy = float(np.median(flow[..., 1])) / gray.shape[0]
                magnitude = float(np.mean(np.linalg.norm(flow, axis=-1))) / diag
                motions.append(
                    CameraMotion(t=idx / fps, dx=dx, dy=dy, motion_magnitude=magnitude)
                )
            prev_gray = gray
        ok, frame = cap.read()
        idx += 1
    cap.release()
    return motions


def motion_blobs(prev_frame: np.ndarray, curr_frame: np.ndarray, top_k: int = 3) -> list[BBox]:
    """Bounding boxes (normalized) of the top-k strongest moving regions between two
    frames, via frame differencing. Used to seed SAM2 tracking on whatever's actually
    moving, independent of Claude's keyframe captions."""
    h, w = curr_frame.shape[:2]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    diff = cv2.GaussianBlur(diff, (9, 9), 0)
    _, mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 0.0005 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        boxes.append((area, BBox(cx=(x + bw / 2) / w, cy=(y + bh / 2) / h, w=bw / w, h=bh / h)))

    boxes.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in boxes[:top_k]]
