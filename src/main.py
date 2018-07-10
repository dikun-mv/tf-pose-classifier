import cv2 as cv
import tensorflow as tf

from time import time
from tf_pose import Estimator
from utils import draw_fps, draw_points

if __name__ == '__main__':
    stream = cv.VideoCapture(0)
    stream.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
    stream.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

    estimator = Estimator(
        '../models/mobilenet.pb',
        (656, 368),
        tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.2))
    )

    while True:
        start_time = time()

        ret, frame = stream.read()
        if not ret: break

        poses = estimator.inference(frame, upsample_size=4.0)

        # frame = estimator.draw_humans(frame, poses)
        frame = draw_points(frame, poses)
        frame = draw_fps(frame, 'FPS: {}'.format(int(1.0 / (time() - start_time))))

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'): break

    stream.release()
    cv.destroyAllWindows()

