import typing as tp
import numpy as np

from abstract_classes.gesture_repr import GestureRepr
from gesture_repr.angles_repr.utils import PointsToAnglesTransformer
from gesture_repr.points_repr.points_repr import PointsRepr


class AnglesRepr(GestureRepr):
    """Extracts and stores key angles from some point_repr using transformer"""

    def __init__(self, points_repr: PointsRepr, transformer: PointsToAnglesTransformer):
        self.angles = np.empty((len(points_repr), len(transformer)))
        # self.angles.shape
        for i in range(len(points_repr)):
            self.angles[i] = transformer.transform(points_repr[i])

    def __len__(self) -> int:
        return len(self.angles)

    def __getitem__(self, ind: tp.Any) -> np.array:
        return self.angles[ind]

    @GestureRepr.shape.getter
    def shape(self) -> tp.Sequence[int]:
        return self.angles.shape

    @property
    def values(self):
        return self.angles

    @values.setter
    def values(self, val: tp.Any):
        self.angles = val
