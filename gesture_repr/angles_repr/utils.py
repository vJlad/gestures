import numpy as np
import typing as tp


class PointsToAnglesTransformer:
    """
    Transforms cnt_point points with to some angles between them.

    angle_description: ( (first_from, first_to), (second_from, second_to) ),
        where params are indexes in [0, cnt_points)
        example of such angle description at 'angle_description_config.py'
    """

    def __init__(self, angle_descriptions: tp.Sequence[tp.Tuple[tp.Tuple[int, int], tp.Tuple[int, int]]] = tuple(),
                 cnt_points: int = 21) -> None:
        self.cnt_points = cnt_points
        self.angle_descriptions = []
        for desc in angle_descriptions:
            self.add_angle_description(*desc[0], *desc[1])

    def add_angle_description(self, first_from: int, first_to: int, second_from: int, second_to: int) -> None:
        assert all(0 <= index < self.cnt_points for index in [first_from, first_to, second_from, second_to]), \
            'wrong index of point'
        self.angle_descriptions.append(((first_from, first_to), (second_from, second_to)))

    def __len__(self, ) -> int:
        return len(self.angle_descriptions)

    def transform(self, points: tp.Sequence) -> np.array:
        assert len(points) == self.cnt_points, 'wrong count of points'
        return np.array([angle_between(points[first_to] - points[first_from],
                                       points[second_to] - points[second_from])
                         for (first_from, first_to), (second_from, second_to) in self.angle_descriptions])


def unit_vector(vector: tp.Sequence) -> tp.Sequence[float]:
    """ Returns the unit vector of the vector. """
    norm = np.linalg.norm(vector)
    assert norm > 0.0, f"Invalid vector norm. |{vector}| = {norm}"
    return np.array(vector) / norm


def angle_between(first: tp.Sequence, second: tp.Sequence) -> float:
    """ Returns the angle in radians between vectors 'first' and 'second'
        If any coord of the input vectors is np.nan, function will return nan
    """
    if np.isnan(first).any() or np.isnan(second).any():
        return np.nan
    return np.arccos(np.clip(np.dot(unit_vector(first), unit_vector(second)), -1.0, 1.0))
