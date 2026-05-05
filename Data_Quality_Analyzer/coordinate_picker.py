import matplotlib as mpl
from matplotlib.backend_tools import ToolToggleBase
import matplotlib.pyplot as plt
import numpy as np


class PickCoords(ToolToggleBase):
    default_keymap = 'y'
    description = 'pick landmarks'
    default_toggled = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coords_dict = {}
        self.background = None

    def enable(self, *args):
        tm = self.figure.canvas.manager.toolmanager
        for tool_name in ['pan', 'zoom']:
            if tool_name in tm.tools:
                t = tm.get_tool(tool_name)
                if t.toggled:
                    print(f"Disabling {tool_name} tool.")
                    tm.trigger_tool(tool_name)

        self.figure.canvas.draw()
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        self.figure.canvas.set_cursor(mpl.backend_tools.Cursors.SELECT_REGION)
        self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def disable(self, *args):
        self.figure.canvas.set_cursor(mpl.backend_tools.Cursors.POINTER)
        self.figure.canvas.mpl_disconnect(self.cid)

    def on_click(self, event):
        if event.inaxes and event.button == 1:
            ax = event.inaxes
            label = ax.get_title()

            if label not in self.coords_dict:
                self.coords_dict[label] = []

            self.coords_dict[label].append((event.xdata, event.ydata))

            self.figure.canvas.restore_region(self.background)

            for active_label, pts in self.coords_dict.items():
                target_ax = [a for a in self.figure.axes if a.get_title() == active_label][0]
                for px, py in pts:
                    dot = target_ax.plot(px, py, 'x', markersize=8, color='red')[0]
                    target_ax.draw_artist(dot)

            self.figure.canvas.blit(self.figure.bbox)


def _select_channel(image, channel=None):
    arr = np.asarray(image, dtype=np.float32)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if channel is None:
            # Default: collapse channels by maximum intensity. For explicit channel
            # alignment, pass channel=<index> from the call site.
            return np.nanmax(arr, axis=-1).astype(np.float32)
        return arr[:, :, int(channel)].astype(np.float32)

    raise ValueError(f"Expected a 2D image or 3D image stack. Got shape {arr.shape}.")


def _normalise01(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)

    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)

    out = np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    vmin = float(vals.min())
    vmax = float(vals.max())

    if vmax <= vmin:
        return out

    out[finite] = (arr[finite] - vmin) / (vmax - vmin)
    return out


def _gradient_magnitude(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    gy, gx = np.gradient(arr)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def peak_gradient_coordinate(
    image,
    channel=None,
    peak_fraction=0.95,
    gradient_center_weight=0.35,
    window_radius=None,
):
    """
    Pick one coordinate from an image using global maximum + local gradient structure.

    The global maximum defines the cell/global high-point neighborhood. Within that
    neighborhood, a weighted centroid is computed over the near-maximum plateau.
    Pixels with lower gradient magnitude are weighted slightly higher, so the chosen
    point lands near the apex/center of the high-intensity region rather than on a
    noisy ridge or an arbitrary plateau edge.

    Returns a dictionary with:
        coord: refined (x, y) coordinate
        max_coord: raw global-maximum (x, y) coordinate
        gradient_at_coord: gradient magnitude near the selected coordinate
    """
    arr = _select_channel(image, channel=channel)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if arr.ndim != 2:
        raise ValueError("Selected image must be 2D.")

    if arr.size == 0 or float(arr.max()) <= 0.0:
        return {
            "coord": (0.0, 0.0),
            "max_coord": (0.0, 0.0),
            "gradient_at_coord": 0.0,
            "peak_value": 0.0,
        }

    max_flat = int(np.argmax(arr))
    max_y, max_x = np.unravel_index(max_flat, arr.shape)
    peak_value = float(arr[max_y, max_x])

    if window_radius is None:
        # Keep the refinement local to the dominant peak while avoiding a single
        # noisy-pixel decision. This is intentionally image-size based, not sigma-based.
        window_radius = max(2, int(round(0.04 * min(arr.shape))))

    window_radius = int(max(1, window_radius))

    y0 = max(0, max_y - window_radius)
    y1 = min(arr.shape[0], max_y + window_radius + 1)
    x0 = max(0, max_x - window_radius)
    x1 = min(arr.shape[1], max_x + window_radius + 1)

    local = arr[y0:y1, x0:x1]
    local_norm = _normalise01(local)
    local_grad = _normalise01(_gradient_magnitude(local))

    threshold = float(np.clip(peak_fraction, 0.0, 1.0))
    near_peak = local >= (threshold * peak_value)

    if not np.any(near_peak):
        near_peak = local == peak_value

    yy, xx = np.nonzero(near_peak)

    intensity_weight = np.maximum(local_norm[yy, xx], 1e-6)
    gradient_center_weight = float(max(0.0, gradient_center_weight))
    gradient_weight = 1.0 + gradient_center_weight * (1.0 - local_grad[yy, xx])
    weights = intensity_weight * gradient_weight

    if float(weights.sum()) <= 0.0:
        refined_y = float(max_y)
        refined_x = float(max_x)
    else:
        refined_y = float(y0 + np.average(yy, weights=weights))
        refined_x = float(x0 + np.average(xx, weights=weights))

    iy = int(np.clip(round(refined_y), 0, arr.shape[0] - 1))
    ix = int(np.clip(round(refined_x), 0, arr.shape[1] - 1))
    grad_full = _gradient_magnitude(arr)

    return {
        "coord": (refined_x, refined_y),
        "max_coord": (float(max_x), float(max_y)),
        "gradient_at_coord": float(grad_full[iy, ix]),
        "peak_value": peak_value,
    }


def pick_peak_gradient_coordinates(
    canvas_img,
    target_img,
    channel=None,
    peak_fraction=0.95,
    gradient_center_weight=0.35,
    window_radius=None,
):
    """
    Pick alignment coordinates from simulation canvas and target MSI.

    This is the automatic path intended for alignment:
        - Simulation_Canvas coordinate: dominant generated high point.
        - Target_Data coordinate: dominant target high point.

    Both coordinates are selected with the same global-maximum + gradient-aware
    refinement rule.
    """
    canvas_pick = peak_gradient_coordinate(
        canvas_img,
        channel=channel,
        peak_fraction=peak_fraction,
        gradient_center_weight=gradient_center_weight,
        window_radius=window_radius,
    )
    target_pick = peak_gradient_coordinate(
        target_img,
        channel=channel,
        peak_fraction=peak_fraction,
        gradient_center_weight=gradient_center_weight,
        window_radius=window_radius,
    )

    return {
        "Simulation_Canvas": [canvas_pick["coord"]],
        "Target_Data": [target_pick["coord"]],
        "diagnostics": {
            "Simulation_Canvas": canvas_pick,
            "Target_Data": target_pick,
        },
    }


def pick_coordinates(
    canvas_img,
    target_img=None,
    *,
    mic_img=None,
    channel=None,
    auto=False,
    peak_fraction=0.95,
    gradient_center_weight=0.35,
    window_radius=None,
    show=True,
):
    """
    Pick coordinates manually or automatically.

    New target-aware call:
        pick_coordinates(generated_canvas, target_msi, auto=True)

    Backward-compatible manual call:
        pick_coordinates(ms_img, mic_img=<microscopy_image>)

    If auto=True, the returned coordinates are derived from the canvas and target
    global maxima plus local gradient structure. If show=True, the picked points
    are displayed for inspection.
    """
    if auto:
        if target_img is None:
            raise ValueError("target_img is required when auto=True.")

        coords = pick_peak_gradient_coordinates(
            canvas_img=canvas_img,
            target_img=target_img,
            channel=channel,
            peak_fraction=peak_fraction,
            gradient_center_weight=gradient_center_weight,
            window_radius=window_radius,
        )

        if show:
            _plot_auto_picks(canvas_img, target_img, coords, channel=channel)

        return coords

    # Manual mode. If target_img is provided, show canvas/target. If not, keep the
    # old behavior and show canvas/microscopy.
    right_img = target_img if target_img is not None else mic_img
    right_title = "Target_Data" if target_img is not None else "Microscopy_Data"

    if right_img is None:
        raise ValueError("Provide target_img or mic_img for manual coordinate picking.")

    plt.rcParams['toolbar'] = 'toolmanager'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    canvas_2d = _select_channel(canvas_img, channel=channel)
    right_2d = _select_channel(right_img, channel=channel)

    ax1.imshow(canvas_2d, cmap='viridis', vmax=np.percentile(canvas_2d, 99))
    ax1.set_title("Simulation_Canvas")

    right_cmap = 'viridis' if target_img is not None else 'gray'
    ax2.imshow(right_2d, cmap=right_cmap, vmax=np.percentile(right_2d, 99))
    ax2.set_title(right_title)

    tm = fig.canvas.manager.toolmanager
    tm.add_tool('PickCoords', PickCoords)
    fig.canvas.manager.toolbar.add_tool('PickCoords', 'navigation')

    plt.tight_layout()
    plt.show()

    return tm.get_tool('PickCoords').coords_dict


def _plot_auto_picks(canvas_img, target_img, coords, channel=None):
    canvas_2d = _select_channel(canvas_img, channel=channel)
    target_2d = _select_channel(target_img, channel=channel)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(canvas_2d, cmap='viridis', vmax=np.percentile(canvas_2d, 99))
    ax1.set_title("Simulation_Canvas")
    for x, y in coords["Simulation_Canvas"]:
        ax1.plot(x, y, 'x', markersize=8, color='red')

    ax2.imshow(target_2d, cmap='viridis', vmax=np.percentile(target_2d, 99))
    ax2.set_title("Target_Data")
    for x, y in coords["Target_Data"]:
        ax2.plot(x, y, 'x', markersize=8, color='red')

    plt.tight_layout()
    plt.show()
