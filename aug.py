import cv2 as cv
import numpy as np


def to_binary_image(image):
    """
        문자 이미지를 처리하기 적합한 이진 이미지 형태로 바꾼다.

        입력 문자 이미지는, 배경이 흰색 / 글자가 검정색 이다.
        입력 문자 이미지는 3 channel, 을 가진다고 가정한다.
        이진 이미지는 편의상 배경이 0 값을 가지도록 변환한다.
    """
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.threshold(image, 100, 255, cv.THRESH_BINARY_INV)[1]
    return image


def to_output_image(b_image):
    """
        이진 이미지를 3 channel 이미지로 변환한다.

        학습에 적합한 float 데이터 타입으로 변환한다.
    """
    b_image[b_image != 0] = 255
    b_image = cv.bitwise_not(b_image)
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


def morphological_transform(b_image, kernel_size):
    """
        이미지에 형태학적 변환을 수행한다.

        m_type 0: morphological erosion (확장)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # morphological dilation
    b_image = cv.dilate(b_image, kernel)

    return b_image


def resizing(b_image, resize_rate):
    """
        이미지를 작은 크기로 줄였다가 다시 원래 크기로 복구시킨다.

        원본 이미지는 30 이상의 너비 / 높이를 가지고있어야 한다.
    """
    height, width = b_image.shape[:2]
    h = int(height*resize_rate)
    h = 30 if h < 30 else h

    w = int(width*resize_rate)
    w = 30 if w < 30 else w

    b_image = cv.resize(b_image, (w, h), interpolation=cv.INTER_NEARES)
    b_image = cv.resize(b_image, (width, height), interpolation=cv.INTER_AREA)

    return b_image


def shearing(b_image, shear_weight):
    """
        이미지를 수평으로 전단시킵니다.
    """
    height, width = b_image.shape[:2]

    # 전단 행렬
    linear_transform = np.float32([[1, shear_weight, 0], [0, 1, 0]])

    # 전단 변환
    adjusted_width = width + abs(int(height * shear_weight))
    linear_transform[0, 2] = -1 * linear_transform[0, 1] * ((width - height * shear_weight) / 2)
    linear_transform[1, 2] = -1 * linear_transform[1, 0] * (height / 2)
    b_image = cv.warpAffine(b_image, linear_transform, (adjusted_width, height))

    # 재조정
    b_image, _, _ = cropping(b_image)
    b_image = cv.resize(b_image, (width, height))

    return b_image


def noising(b_image, level):
    """
        이미지에 무작위 노이즈를 생성하고 보정합니다.
    """
    size = b_image.flatten().shape[0]
    random_value = np.random.uniform(0, 1, size)
    noise = np.zeros(random_value.shape, dtype=np.uint8)

    noise[random_value <= level] = 255
    noise = noise.reshape(b_image.shape)

    b_image = cv.bitwise_xor(b_image, noise)
    return b_image


def random_morphological_transform(b_image, level, kernel, m_type):
    """
        이미지를 무작위 필터로 morphological transform 적용합니다.

        m_type == 0: morphological erosion
        m_type == 1: morphological dilation
    """
    kernel_size = kernel.shape[0]

    result = b_image.copy()
    h_size = b_image.shape[0] - kernel_size + 1
    w_size = b_image.shape[1] - kernel_size + 1

    # Vectorized Solution - using numpy
    # get window
    window_shape = (h_size, w_size) + kernel.shape
    window_strides = b_image.strides + b_image.strides
    window = np.lib.stride_tricks.as_strided(b_image, window_shape, window_strides)
    window = window.reshape((-1, kernel_size, kernel_size))

    # get random window
    rnd = np.random.uniform(0, 1, window.shape[0])
    index = np.argwhere(rnd < level)
    window = window[index].reshape((-1, kernel_size, kernel_size))

    if m_type == 0:
        # morphological erosion
        kernel = np.bitwise_not(kernel)
        masked_window = np.bitwise_or(window, kernel)
        chk = np.all(masked_window != 0, axis=1)
        chk = np.all(chk != 0, axis=1)
        adj = np.zeros(shape=(chk.shape[0], 1), dtype=np.uint8)
        adj[chk] = 1
    else:
        # morphological dilation
        masked_window = np.bitwise_and(window, kernel)
        chk = np.bitwise_not(np.all(masked_window == 0, axis=1))
        chk = np.all(chk == 0, axis=1)
        adj = np.ones(shape=(chk.shape[0], 1), dtype=np.uint8)
        adj[chk] = 0

    # adjust
    adj = adj * 255
    k = int(kernel_size / 2)
    h = (index // w_size).astype(np.int) + k
    w = np.mod(index, w_size).astype(np.int) + k

    result[h, w] = adj

    """
    # loop 이용... 너무 느리다.
    k = int(kernel_size / 2)
    rnd_idx = -1

    for h in range(0, h_size):
        for w in range(0, w_size):
            rnd_idx = rnd_idx + 1
            if rnd_check[rnd_idx]:
                # random skip
                continue

            window = b_image[h:h+kernel_size, w:w+kernel_size]
            window = cv.bitwise_and(window, kernel)

            if m_type == 0:
                # morphological erosion
                adj = 255 if np.all(window != 0) else 0
            else:
                # morphological dilation
                adj = 0 if np.all(window == 0) else 255

            result[k + h, k + w] = adj
    """

    return result


class AugmentationGenerator:
    ORIGINAL_RATE = 0.2
    RANDOM_MORPHOLOGICAL_TRANSFORM_PROBABILITY = 0.0
    RESIZING_PROBABILITY = 0.0
    SHEARING_PROBABILITY = 0.0
    MEDIAN_BLURRING_PROBABILITY = 0.0
    NOISING_PROBABILITY = 0.0

    def __init__(self, seed=None):
        if seed is None:
            self.seed = 0
        else:
            self.seed = seed
        np.random.seed(seed)

    def randomize(self, image, width, height):
        # 이미지를 이진 이미지로 변환
        b_image = to_binary_image(image)

        # Cropping
        b_image, _, _ = cropping(b_image)

        p = np.random.uniform(0, 1, 5)
        # Original
        if p[0] < self.ORIGINAL_RATE:
            b_image = cv.resize(b_image, (width, height))
            b_image = to_output_image(b_image)
            return b_image

        # Shearing
        if p[1] < self.SHEARING_PROBABILITY:
            shear_weight = np.random.uniform(0, 1, 1) - 0.5
            shear_weight = shear_weight * 1.6  # shear_weight 는 [-0.8 ~ 0.8]
            b_image = shearing(b_image, shear_weight)

        # Random Morphological Transform
        if p[2] < self.RANDOM_MORPHOLOGICAL_TRANSFORM_PROBABILITY:
            kernel = np.ones((3, 3), dtype=np.uint8) * 255
            b_image = random_morphological_transform(b_image, 0.5, kernel, 1)
            b_image = random_morphological_transform(b_image, 0.5, kernel, 0)

        # Median Blurring
        if p[3] < self.MEDIAN_BLURRING_PROBABILITY:
            kernel_size = np.random.randint(1, 3, 1)
            kernel_size = kernel_size[0] * 2 + 1

            b_image = cv.medianBlur(b_image, kernel_size)

        # Character Resize
        scale = np.random.uniform(0.7, 1, 1)
        w = int(width * scale)
        h = int(height * scale)
        b_image = cv.resize(b_image, (w, h))

        # Random Padding
        x_length = width - w
        y_length = height - h

        left = int(x_length / 2)
        right = left + x_length % 2
        top = int(y_length / 2)
        bottom = top + y_length % 2

        if left > 0:
            random_dx = np.random.randint(-left, left)
            left = left - random_dx
            right = right + random_dx

        if top > 0:
            random_dy = np.random.randint(-top, top)
            top = top - random_dy
            bottom = bottom + random_dy

        b_image = cv.copyMakeBorder(b_image, top, bottom, left, right, cv.BORDER_CONSTANT, 0)

        # Noising
        if p[4] < self.NOISING_PROBABILITY:
            noise_level = np.random.uniform(0.05, 0.15, 1)[0]
            b_image = noising(b_image, noise_level)

        # Return
        b_image = to_output_image(b_image)

        return b_image

