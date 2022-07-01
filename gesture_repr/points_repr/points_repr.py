import numpy as np
import mediapipe as mp
import typing as tp

from abstract_classes.gesture_repr import GestureRepr


class PointsRepr(GestureRepr):
    """
    Extracts and stores key-points from frames of video.
    Uses MediaPipe -> 21 3d float points for each frame (or np.nan)
    """

    def __init__(self, video_frames: tp.Sequence):
        video_frames = np.array(video_frames)
        assert len(video_frames.shape) == 4, f"Video is a sequence of frames, expected" \
                                             f"4(cnt_frames, h, w, rgb) dimensions, got {len(video_frames.shape)=} " \
                                             f"dimensions"
        self.points = np.empty((video_frames.shape[0], 21, 3), dtype='float32')
        with mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.5) as hands:
            for i, frame in enumerate(video_frames):
                res = hands.process(frame)
                self.points[i] = (np.array([[landmark.x, landmark.y, landmark.z]
                                            for landmark in res.multi_hand_landmarks[0].landmark])
                                  if res.multi_hand_landmarks is not None
                                  else np.array([[np.nan, np.nan, np.nan]] * 21))

    def __len__(self) -> int:
        return len(self.points)

    def __getitem__(self, ind: tp.Any) -> np.array:
        return self.points[ind]

    @GestureRepr.shape.getter
    def shape(self) -> tp.Sequence[int]:
        return self.points.shape

    @property
    def values(self):
        return self.points

    @values.setter
    def values(self, val: tp.Any):
        self.points = val