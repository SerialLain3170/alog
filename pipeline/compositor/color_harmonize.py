import cv2
import numpy as np


def harmonize_color(
    char_bgr: np.ndarray, char_alpha: np.ndarray, bg_crop_bgr: np.ndarray, strength: float = 0.35
) -> np.ndarray:
    """Partially matches `char_bgr`'s color statistics (mean/std in LAB space) to
    `bg_crop_bgr`'s — a Reinhard-style transfer — blended at `strength` (0 = no
    change, 1 = full statistical match) so the character keeps its own identifiable
    colors while looking less pasted-on against the scene's lighting/color grade.
    `char_alpha` selects which char_bgr pixels count toward its own statistics —
    fully transparent background pixels would otherwise skew them."""
    if strength <= 0:
        return char_bgr

    mask = char_alpha > 0
    if not mask.any() or bg_crop_bgr.size == 0:
        return char_bgr

    char_lab = cv2.cvtColor(char_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    char_mean = char_lab[mask].mean(axis=0)
    char_std = char_lab[mask].std(axis=0) + 1e-6
    bg_mean = bg_lab.reshape(-1, 3).mean(axis=0)
    bg_std = bg_lab.reshape(-1, 3).std(axis=0) + 1e-6

    transferred = (char_lab - char_mean) * (bg_std / char_std) + bg_mean
    blended_lab = np.clip(char_lab * (1 - strength) + transferred * strength, 0, 255).astype(np.uint8)

    return cv2.cvtColor(blended_lab, cv2.COLOR_LAB2BGR)
