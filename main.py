"""Enhance Image Using Tensorflow"""
import cv2
import numpy as np
import os


def resizeImage(output: bool, img: np.ndarray):
    """Resizing the image to either 215x215 or 860x860
        :param
        output: if True resize output to 860 if no to 215"""
    if output:
        return cv2.resize(img, (860, 860), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (215, 215), interpolation=cv2.INTER_AREA)

