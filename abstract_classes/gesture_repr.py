from abc import ABC, abstractmethod
import typing as tp


class GestureRepr(ABC):
    """ Abstract class for gesture representation used in algorithms """

    @abstractmethod
    def __init__(self, video_frames: tp.Sequence):
        pass

    @abstractmethod
    def __len__(self, ) -> int:
        pass

    @abstractmethod
    def __getitem__(self, ind: tp.Any) -> tp.Any:
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def values(self):
        pass

    @values.setter
    @abstractmethod
    def values(self, val: tp.Any):
        pass
