from abc import ABC, abstractmethod
from abstract_classes.gesture_repr import GestureRepr


class GestureComparator(ABC):
    @abstractmethod
    def add_valid_gesture(self, gesture_repr: GestureRepr) -> None:
        pass

    @abstractmethod
    def is_valid(self, gesture_repr: GestureRepr) -> bool:
        pass

    @abstractmethod
    def proba_is_valid(self, gesture_repr: GestureRepr) -> float:
        pass