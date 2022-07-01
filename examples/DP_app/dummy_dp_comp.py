from abstract_classes.gesture_comparator import GestureComparator
from gesture_repr.angles_repr.angles_repr import AnglesRepr
from distance.per_series_dist.gesture_dist import gesture_series_dist
from distance.per_series_dist.dp_dist import dp_time_series_distance


class DummyDPComparator(GestureComparator):
    """
    Dummy comparator:
    If for given gesture exists valid gesture with distance <= threshold,
    given gesture considered valid.
    Probability is computed as 0.5**(min_dist)
    """
    def __init__(self, threshold: float = 0.2, **dp_dist_params):
        assert threshold >= 0.0, "gesture dp_dist cannot be less than 0"
        self.threshold = threshold
        self.dp_dist_params = dp_dist_params
        self.valid_gestures = []

    def add_valid_gesture(self, gesture_repr: AnglesRepr) -> None:
        self.valid_gestures.append(gesture_repr)

    def is_valid(self, gesture_repr: AnglesRepr) -> bool:
        for valid_gesture in self.valid_gestures:
            dist = gesture_series_dist(valid_gesture,
                                        gesture_repr,
                                        dp_time_series_distance,
                                        **self.dp_dist_params).max()
            if dist <= self.threshold:
                return True
        return False

    def proba_is_valid(self, gesture_repr: AnglesRepr):
        min_dist = None
        for valid_gesture in self.valid_gestures:
            dist = gesture_series_dist(valid_gesture,
                                        gesture_repr,
                                        dp_time_series_distance,
                                        **self.dp_dist_params).max()
            if min_dist is None or min_dist > dist:
                min_dist = dist
        return 0.5 ** (min_dist / self.threshold)
