"""Staff crop -> fixed-height model input.

Part of the delivered path, not of the training rig: the browser has to produce
byte-identical input. Hence plain numpy again, bilinear resampling written out
by hand -- no PIL, no OpenCV, nothing that has no JS counterpart.

Scaling is driven by the detected staff-line spacing rather than by dpi, so a
page set at any resolution lands on the same geometry.
"""
import numpy as np

TARGET_SPACING = 10.0     # px between staff lines after normalisation
TARGET_HEIGHT = 128       # px, fixed; width stays variable


def resize_bilinear(img, out_h, out_w):
    """Bilinear resample of a float32 (H, W) image."""
    in_h, in_w = img.shape
    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32, copy=False)

    # Half-pixel centres, so the result does not drift by half a pixel.
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (in_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (in_w / out_w) - 0.5
    ys = np.clip(ys, 0, in_h - 1)
    xs = np.clip(xs, 0, in_w - 1)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]

    top = img[y0][:, x0] * (1 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1 - wx) + img[y1][:, x1] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


def normalize_staff(crop, line_spacing, staff_offset=None,
                    target_spacing=TARGET_SPACING, target_height=TARGET_HEIGHT):
    """Scale a staff crop to the canonical spacing and pad it to a fixed height.

    `staff_offset` is the distance from the top of the crop to the topmost staff
    line, so the five lines land in the same place for every sample regardless of
    how far the adaptive padding grew. Without it the crop is simply centred.

    Returns float32 (target_height, W') in [0,1], white = 1.
    """
    scale = target_spacing / float(line_spacing)
    h, w = crop.shape
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    scaled = resize_bilinear(crop.astype(np.float32), out_h, out_w)

    canvas = np.ones((target_height, out_w), dtype=np.float32)

    if staff_offset is None:
        top = (target_height - out_h) // 2
    else:
        # Put the staff itself at a fixed position: four spacings of staff plus
        # equal air above and below.
        staff_top_scaled = staff_offset * scale
        want = (target_height - 4 * target_spacing) / 2.0
        top = int(round(want - staff_top_scaled))

    src_y0 = max(0, -top)
    src_y1 = min(out_h, target_height - top)
    dst_y0 = max(0, top)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    if src_y1 > src_y0:
        canvas[dst_y0:dst_y1, :] = scaled[src_y0:src_y1, :]
    return canvas
