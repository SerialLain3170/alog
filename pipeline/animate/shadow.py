import cv2
import numpy as np


def foot_contact_point(alpha_mask: np.ndarray) -> tuple[int, int, int] | None:
    """Returns (x_center, y_bottom, width) of the character's ground-contact
    footprint from its alpha mask, or None if the mask is empty."""
    ys, xs = np.where(alpha_mask > 0)
    if xs.size == 0:
        return None

    y_bottom = int(ys.max())
    row_xs = np.where(alpha_mask[y_bottom, :] > 0)[0]
    if row_xs.size == 0:
        row_xs = xs
    x_center = int((row_xs.min() + row_xs.max()) / 2)
    width = max(int(row_xs.max() - row_xs.min()), int(xs.max() - xs.min()), 1)
    return x_center, y_bottom, width


def render_shadow(alpha_mask: np.ndarray, background_frame: np.ndarray, base_opacity: float = 0.35) -> np.ndarray:
    """Procedural soft contact shadow: a blurred dark ellipse under the character's
    feet, modulated by local background luminance (dim scenes get a fainter shadow
    since ambient contrast is already low). Returns a single-channel (h, w) float
    array in [0, 1] — darkness to multiply onto the background at compositing time."""
    h, w = alpha_mask.shape
    shadow = np.zeros((h, w), dtype=np.float32)

    contact = foot_contact_point(alpha_mask)
    if contact is None:
        return shadow

    x_center, y_bottom, width = contact
    axes = (max(int(width * 0.55), 4), max(int(width * 0.16), 3))
    cv2.ellipse(shadow, (x_center, y_bottom), axes, 0, 0, 360, color=1.0, thickness=-1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=max(width * 0.08, 2))

    gray_bg = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    y0, y1 = max(y_bottom - width // 4, 0), min(y_bottom + width // 4, h)
    x0, x1 = max(x_center - width, 0), min(x_center + width, w)
    local_luminance = float(gray_bg[y0:y1, x0:x1].mean()) if y1 > y0 and x1 > x0 else 0.5

    return shadow * base_opacity * (0.4 + 0.6 * local_luminance)
