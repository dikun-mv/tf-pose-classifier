import sys
import cv2
import json
import numpy as np

from os import path
from tf_pose import Estimator

from utils import draw_points, poses_to_np

MODEL_PATH = path.realpath(path.join(path.dirname(__file__), '../models/mobilenet.pb'))

if __name__ == '__main__':
    video_path = sys.argv[1]

    if not video_path:
        print('Unspecified video path')
        exit(1)

    input = cv2.VideoCapture(video_path)
    estimator = Estimator(MODEL_PATH, (656, 368))
    p_seq = []

    while True:
        ret, frame = input.read()
        if not ret: break

        poses = estimator.inference(frame, upsample_size=4.0)
        p_seq.append(poses_to_np(poses))

    input.release()

    p_seq = np.array(p_seq[:50])
    v_seq = np.array([p_seq[i + 1] - p_seq[i] for i in range(len(p_seq) - 1)])
    empty = np.tile(.0, (50, 36))
    empty[:len(v_seq)] = v_seq

    with open(video_path.split('.')[0] + '.json', 'w') as output:
        json.dump(empty.tolist(), output)
