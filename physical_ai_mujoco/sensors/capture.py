"""Acquisizione di immagini monocromatiche e, opzionalmente, depth."""

from physical_ai_mujoco.contracts.sensing import StereoFrame


# Distanza fisica, in metri, tra i centri delle due camere nei test stereo.
STEREO_BASELINE_M = 0.15


class SimulatedStereoCamera:
    """Legge solo misure del simulatore; non produce ID o maschere."""

    def __init__(self, *, with_depth: bool = False):
        if not isinstance(with_depth, bool):
            raise TypeError("with_depth deve essere un booleano")
        self.with_depth = with_depth

    def capture(self, simulator) -> StereoFrame:
        import cv2

        left_rgb, right_rgb = simulator.render_stereo()
        left_gray = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2GRAY)
        intrinsics, world_from_left, baseline = simulator.stereo_calibration()
        depth_left = depth_right = None
        if self.with_depth:
            depth_left, depth_right = simulator.render_depth_stereo()
            if depth_left.shape != left_gray.shape or depth_right.shape != right_gray.shape:
                raise ValueError("Depth e immagini stereo hanno dimensioni diverse")
        return StereoFrame(
            cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB),
            intrinsics, world_from_left, baseline,
            simulator.time, "world", depth_left, depth_right,
        )
