import tensorflow as tf
import numpy as np
import cv2 as cv

from os import path
from tf_pose import Estimator
from cam_reader import Reader

TF_CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
MODEL_PATH = path.realpath(path.join(path.dirname(__file__), '../models/mobilenet.pb'))


class RingBuffer:
    def __init__(self, size):
        self.head = 0
        self.size = size
        self.data = [None for _ in range(size)]

    def save(self, item):
        if self.head == self.size:
            self.head = 0

        self.data[self.head] = item
        self.head += 1

    def dump(self):
        return self.data


if __name__ == '__main__':
    reader = Reader(1920, 1080)
    estimator = Estimator(MODEL_PATH, (368, 368), TF_CONFIG)
    buffer = RingBuffer(100)

    frame_number = 0

    while True:
        frame = reader.read()
        frame_number += 1

        if frame is None: break

        poses = estimator.inference(frame, upsample_size=4.0)
