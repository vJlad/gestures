import cv2
import typing as tp
import numpy as np

from abstract_classes.controller import Controller
from examples.DP_app.dummy_dp_comp import DummyDPComparator
from gesture_repr.points_repr.points_repr import PointsRepr
from gesture_repr.angles_repr.angles_repr import AnglesRepr
from gesture_repr.angles_repr.utils import PointsToAnglesTransformer
from gesture_repr.repr_modify_utils.utils import erase_nan_prefix_suffix, interpolate, nan_percentage
from smoothing.smooth_gesture import smooth_gesture
from examples.DP_app.angle_description_config import angles_description


class DummyController(Controller):
    def __init__(self):
        self.comparator = DummyDPComparator(0.2,
                                            max_erases=lambda x, y: abs(x - y) + int(min(x, y) * 0.015
                                                                                     * np.log(min(x, y) + 1)))

        self.angle_transformer = PointsToAnglesTransformer(angles_description, 21)

    def preproc_video(self, video_cap: cv2.VideoCapture) -> tp.Optional[AnglesRepr]:
        all_frames = []
        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break
            all_frames.append(frame)
        all_frames = np.array(all_frames)
        angle_repr = PointsRepr(all_frames)
        angle_repr = AnglesRepr(angle_repr, self.angle_transformer)
        angle_repr.values = erase_nan_prefix_suffix(angle_repr.values)
        if angle_repr is None or nan_percentage(angle_repr) > 0.15:
            return None
        angle_repr = interpolate(angle_repr)
        angle_repr = smooth_gesture(angle_repr, return_last_gpr_mean_std=True)
        angle_repr = angle_repr[:, 1, :].T
        return angle_repr

    def add_valid_gesture(self, video_cap: cv2.VideoCapture) -> bool:
        angle_repr = self.preproc_video(video_cap)
        if angle_repr is None:
            return False
        self.comparator.add_valid_gesture(angle_repr)
        return True

    def is_valid(self, video_cap: cv2.VideoCapture) -> bool:
        angle_repr = self.preproc_video(video_cap)
        if angle_repr is None:
            return False
        return self.comparator.is_valid(angle_repr)

    def proba_is_valid(self, video_cap: cv2.VideoCapture) -> float:
        angle_repr = self.preproc_video(video_cap)
        if angle_repr is None:
            return 0.0
        return self.comparator.proba_is_valid(angle_repr)
