import tensorflow as tf

from os import path
from time import time, sleep
from tf_pose import Estimator
from ws_publisher import Publisher, poses_to_dto
from cam_reader import Reader

TF_CONFIG = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
MODEL_PATH = path.realpath(path.join(path.dirname(__file__), '../models/mobilenet.pb'))

if __name__ == '__main__':
    reader = Reader(1920, 1080)
    publisher = Publisher('0.0.0.0', 9090)
    estimator = Estimator(MODEL_PATH, (656, 368), TF_CONFIG)

    start_time = time()
    frame_number = 0

    while True:
        frame_number += 1
        frame = reader.read()

        if frame is None: break

        poses = estimator.inference(frame, upsample_size=4.0)
        publisher.send(poses_to_dto(poses))

        sleep(5 / 1000)

        elapsed_time = time() - start_time

        if elapsed_time >= 1:
            print('FPS: {}'.format(int(frame_number / elapsed_time)))

            start_time = time()
            frame_number = 0
