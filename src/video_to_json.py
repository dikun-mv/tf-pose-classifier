import sys
import cv2
import json

from os import path
from tf_pose import Estimator

from utils import draw_points

MODEL_PATH = path.realpath(path.join(path.dirname(__file__), '../models/mobilenet.pb'))
FRAME_SIZE = (160, 120)

if __name__ == '__main__':
    video_path = sys.argv[1]

    if not video_path:
        print('Unspecified video path')
        exit(1)

    input = cv2.VideoCapture(video_path)
    estimator = Estimator(MODEL_PATH, FRAME_SIZE)
    pose_history = []

    while True:
        ret, frame = input.read()
        if not ret: break

        poses = estimator.inference(frame, upsample_size=4.0)
        pose_history.append(
            [[poses[0].body_parts[i].x, poses[0].body_parts[i].y] if i in poses[0].body_parts else [0., 0.] for i in
             range(18)] if poses else [[0., 0.] for i in range(18)])

        cv2.imshow('main', draw_points(frame, poses))
        cv2.waitKey(1)

    input.release()

    with open(video_path.split('.')[0] + '.json', 'w') as output:
        json.dump(pose_history, output)
