"""
Capture of webcam and manipulation (flipping & rotating)

Contains test code if run as main
"""

import cv2 as cv

_DEBUG_WINDOW_NAME = "Debug: Raw Input"
_FLIP_CODES = (None, 0, 1, -1)
_FLIP_NAMES = ("None", "Horizontal", "Vertical", "Both")
_ROTATE_CODES = (None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE)
_ROTATE_NAMES = ("None", "90°", "180°", "270°")


class Webcam:
    """
    Use as an iterator to get frame-by-frame video.
    Wrapper around the openCV code for ease of use.
    """
    def __init__(self, flip=None, rotate=None, show_debug=False):
        self.flip = flip
        self.rotate = rotate
        self.show_debug = show_debug
        self.cap: cv.VideoCapture = cv.VideoCapture(0)
        if show_debug:
            cv.namedWindow(_DEBUG_WINDOW_NAME, cv.WINDOW_AUTOSIZE | cv.WINDOW_KEEPRATIO | cv.WINDOW_GUI_NORMAL)

    def __del__(self):
        self.cap.release()

    def set_resolution(self, w, h):
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    def __iter__(self):
        return self

    def __next__(self):
        ret, raw_frame = self.cap.read()
        if not ret:
            raise RuntimeError("No more frames. (Webcam unplugged?)")
        if self.flip is not None:
            raw_frame = cv.flip(raw_frame, self.flip)
        if self.rotate is not None:
            raw_frame = cv.rotate(raw_frame, self.rotate)
        if self.show_debug:
            cv.imshow(_DEBUG_WINDOW_NAME, raw_frame)
        return raw_frame

    def toggle_flip(self):
        i = _FLIP_CODES.index(self.flip) + 1
        i %= len(_FLIP_CODES)
        self.flip = _FLIP_CODES[i]
        return _FLIP_NAMES[i]

    def toggle_rotate(self):
        i = _ROTATE_CODES.index(self.rotate) + 1
        i %= len(_ROTATE_CODES)
        self.rotate = _ROTATE_CODES[i]
        return _ROTATE_NAMES[i]


if __name__ == '__main__':
    webcam = Webcam(show_debug=True)
    webcam.set_resolution(1280, 800)
    for frame in webcam:
        cv.waitKey(1)
