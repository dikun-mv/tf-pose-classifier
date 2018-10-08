import tensorflow as tf
import numpy as np

from os import path
from tf_pose import Estimator
from cam_reader import Reader
from keras.models import load_model

TF_CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
MODEL_PATH = path.realpath(path.join(path.dirname(__file__), '../models/mobilenet.pb'))
CLASSIFIER_PATH = path.realpath(path.join(path.dirname(__file__), '../models/pose-classifier.h5'))
BUFFER_SIZE = 100


def poses_to_np(poses):
    return np.array(
        [[poses[0].body_parts[i].x, poses[0].body_parts[i].y] if i in poses[0].body_parts else [0., 0.] for i in
         range(18)] if poses else [[0., 0.] for i in range(18)]).flatten()


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
    classifier = load_model(CLASSIFIER_PATH)
    buffer = RingBuffer(BUFFER_SIZE)

    frame_number = 0

    while True:
        frame = reader.read()
        frame_number += 1

        if frame is None: break

        poses = estimator.inference(frame, upsample_size=4.0)
        buffer.save(poses_to_np(poses))

        if frame_number > BUFFER_SIZE:
            classes = classifier.predict(np.array(buffer.dump()).reshape((1, 100, 36)))

            for i in range(len(classes)):
                idx = np.argmax(classes[i])
                print('{}: {}'.format(idx, classes[i][idx]))
