import numpy as np
import typing as tp
import pandas as pd
from copy import deepcopy

from abstract_classes.gesture_repr import GestureRepr


def erase_nan_prefix_suffix(gesture_repr: GestureRepr) -> tp.Optional[GestureRepr]:
    """Erases lines with at least one np.nan on prefix and suffix"""
    left, right = 0, gesture_repr.shape[0]
    while left < right and np.isnan(gesture_repr[left:left + 1, :]).any():
        left += 1
    while left < right and np.isnan(gesture_repr[right - 1:right, :]).any():
        right -= 1
    if left == right:
        return None
    return gesture_repr[left:right, :]


def nan_percentage(gesture_repr: GestureRepr) -> float:
    """Calculates percentage of lines with at least one np.nan"""
    return np.isnan(gesture_repr).any(axis=1).sum() / gesture_repr.shape[0]


def interpolate(gesture_repr: GestureRepr) -> GestureRepr:
    """Interpolates every np.nan by columns, supposes that gesture_repr.values is 2d np.array"""
    ans = deepcopy(gesture_repr)
    df = pd.DataFrame(ans.values)
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    ans.values = df.values
    return ans
