import cv2 as cv
import numpy as np


def char_to_binary_image(image):
    """
        문자 이미지를 처리하기 적합한 이진 이미지 형태로 바꾼다.

        입력 문자 이미지는, 배경이 흰색 / 글자가 검정색 이다.
        입력 문자 이미지는 3 channel, 을 가진다고 가정한다.
        이진 이미지는 편의상 배경이 0 값을 가지도록 변환한다.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.threshold(image, 100, 255, cv.THRESH_BINARY_INV)[1]
    return image


def binary_image_to_char(b_image):
    """
        이진 이미지를 3 channel 이미지로 변환한다.

        학습에 적합한 float 데이터 타입으로 변환한다.
    """
    image = cv.cvtColor(b_image, cv.COLOR_GRAY2BGR)
    image = image.astype(np.float64)
    return image


def cropping(b_image):
    """
        이미지의 글자부분에 알맞도록 배경을 자른다.

        수정된 이미지와 함께 원본 이미지의 크기를 반환한다.
    """
    height, width = b_image.shape[:2]
    x_flat = np.sum(b_image, axis=0) != 0
    y_flat = np.sum(b_image, axis=1) != 0

    left = x_flat.argmax()
    right = width - x_flat[::-1].argmax()
    top = y_flat.argmax()
    bottom = height - y_flat[::-1].argmax()

    return b_image[top:bottom, left:right], height, width


def padding(b_image, height, width):
    """
        이미지를 지정된 크기까지 확장시킨다.

        인수로 주어진 높이와 너비가 이미지보다 크다고 가정한다.
    """
    h, w = b_image.shape[:2]
    x_length = width - w
    y_length = height - h

    left = int(x_length/2)
    right = left + x_length % 2
    top = int(y_length/2)
    bottom = top + y_length % 2

    b_image = cv.copyMakeBorder(b_image, top, bottom, left, right, cv.BORDER_CONSTANT, 0)
    return b_image


class AugmentationGenerator:
    def __init__(self, seed=None):
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed

    def randomize(self, image, shape):
        pass
