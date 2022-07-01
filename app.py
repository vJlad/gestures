import os
import cv2

from examples.DP_app.dummy_controller import DummyController

if __name__ == '__main__':
    dc = DummyController()
    path1 = "video_examples/1/"
    path2 = "video_examples/2/"
    v1 = [path1 + file for file in os.listdir(path1)]
    v2 = [path2 + file for file in os.listdir(path2)]
    print(f'{v1=}')
    print(f'{v2=}')
    for video_path in v1[:-2]:
        dc.add_valid_gesture(cv2.VideoCapture(video_path))

    for video_path in v1[-2:] + v2:
        print(dc.proba_is_valid(cv2.VideoCapture(video_path)))