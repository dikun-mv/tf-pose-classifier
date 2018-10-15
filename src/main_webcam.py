import tensorflow as tf
import numpy as np

from os import path
from tf_pose import Estimator
from cam_reader import Reader
from keras.models import load_model
from utils import poses_to_np

TF_CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
MODEL_PATH = path.realpath(path.join(path.dirname(__file__), '../models/mobilenet.pb'))
CLASSIFIER_PATH = path.realpath(path.join(path.dirname(__file__), '../models/posec-3cn-conv-5k.h5'))
BUFFER_SIZE = 50


class RingBuffer:
    def __init__(self, size):
        self.head = 0
        self.size = size
        self.data = np.tile(.0, (size, 36))

    def save(self, item):
        if self.head == self.size:
            self.head = 0

        self.data[self.head] = item
        self.head += 1

    def dump(self):
        return self.data


class LinearBuffer:
    def __init__(self, size):
        self.data = np.tile(.0, (size, 36))

    def save(self, item):
        self.data[:-1] = self.data[1:]
        self.data[-1] = item

    def dump(self):
        return self.data


if __name__ == '__main__':
    reader = Reader(1920, 1080)
    estimator = Estimator(MODEL_PATH, (656, 368), TF_CONFIG)
    classifier = load_model(CLASSIFIER_PATH)
    buffer = LinearBuffer(BUFFER_SIZE)

    frame_number = 0

    while True:
        frame = reader.read()
        frame_number += 1

        if frame is None: break

        poses = estimator.inference(frame, upsample_size=4.0)
        buffer.save(poses_to_np(poses))

        p_seq = buffer.dump()
        v_seq = np.array([p_seq[i + 1] - p_seq[i] for i in range(len(p_seq) - 1)])
        empty = np.tile(.0, (BUFFER_SIZE, 36))
        empty[:len(v_seq)] = v_seq

        predictions = classifier.predict(empty.reshape((1, BUFFER_SIZE, 36)))

        idx = np.argmax(predictions[0])
        acc = predictions[0][idx]

        if acc > 0.5: print('{}: {}'.format(idx, acc))
