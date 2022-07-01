import typing as tp
import numpy as np

from abstract_classes.gesture_repr import GestureRepr
from distance.per_series_dist.dp_dist import dp_time_series_distance


def gesture_series_dist(first_gesture_repr: GestureRepr,
                        second_gesture_repr: GestureRepr,
                        time_series_dist: tp.Callable = dp_time_series_distance,
                        **time_series_dist_params
                        ) -> np.array:
    """
    Iterates over time series of each gesture representations (supposes first dim as time),
    computes time series distances.
    :param first_gesture_repr: first input gesture
    :param second_gesture_repr: second input gesture
    :param time_series_dist: Time series distance function
    :param time_series_dist_params: Time series distance function params
    :return: np.array of time series distances
    """
    ans = np.empty((first_gesture_repr.shape[1],))
    for i in range(first_gesture_repr.shape[1]):
        ans[i] = time_series_dist(first_gesture_repr[:, i], second_gesture_repr[:, i], **time_series_dist_params)
    return ans