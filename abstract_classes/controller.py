from abc import ABC, abstractmethod
import typing as tp

from abstract_classes.gesture_repr import GestureRepr


class Controller(ABC):
    @abstractmethod
    def preproc_video(self, video: tp.Any) -> GestureRepr:
        pass

    @abstractmethod
    def add_valid_gesture(self, video: tp.Any) -> None:
        pass

    @abstractmethod
    def is_valid(self, video: tp.Any) -> bool:
        pass

    @abstractmethod
    def proba_is_valid(self, video: tp.Any) -> float:
        pass