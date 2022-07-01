import typing as tp
import numpy as np

from abstract_classes.gesture_repr import GestureRepr
from smoothing.gpr_smoothing import gpr_time_series_smoothing


def smooth_gesture(gesture_repr: GestureRepr,
                   smoothing_func: tp.Callable = gpr_time_series_smoothing,
                   **smoothing_func_params):
    """
    Supposes that gesture_repr is a 2d np.array with first dim --- time
    :param gesture_repr:
        Some gesture representation to be smoothed
    :param smoothing_func:
        Function that smooth each time series of gesture
    :param smoothing_func_params:
        Params for passing to the smoothing_func
    :return:
        Smoothed gesture in the same representation
    """
    ans = np.array([smoothing_func(gesture_repr[:, i], **smoothing_func_params)
                    for i in range(gesture_repr.shape[1])])
    return ans
