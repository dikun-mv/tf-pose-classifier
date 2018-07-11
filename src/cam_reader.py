import cv2 as cv

from threading import Thread


class Reader(Thread):
    def __init__(self, width, height, device=0):
        super().__init__(daemon=True)
        self._stream = cv.VideoCapture(device)
        self._stream.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self._stream.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self._frame = None
        self.start()

    def __del__(self):
        self._frame = None
        self._stream.release()

    def run(self):
        while True:
            ret, frame = self._stream.read()

            if not ret:
                self._frame = None
                break

            self._frame = frame

    def read(self):
        return self._frame
