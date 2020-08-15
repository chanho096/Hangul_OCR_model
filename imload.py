import cv2 as cv
import numpy as np
import gc
import os


def get_package_size(batch_size, image_shape, limit_package_size):
    image_bit_size = image_shape[0] * image_shape[1] * image_shape[2] * 8
    package_size = int(limit_package_size * (1024**3) / image_bit_size)
    package_size = int(package_size / batch_size) * batch_size
    return package_size


def get_package_index(batch_index, batch_size, package_size):
    """
        해당 batch index 가 포함된 package index 를 찾는다.
    """
    index = batch_index * batch_size
    package_index = int(index / package_size)
    package_start = int((index % package_size) / batch_size)

    return package_index, package_start


class ImageDataLoader:
    """
        이미지 데이터를 numpy package 로 저장 후, batch 단위로 읽는다.

        SSD 의 대용량 데이터 고속 전송을 이용하여 학습 속도를 향상시킨다.
    """
    def __init__(self):
        self._package = None  # 현재 적재된 package
        self._current_package = -1  # 현재 적재된 package 의 index

        self._package_dir = None  # package 파일이 저장될 경로
        self._package_count = 0  # 현재 생성된 package 개수
        self._batch_size = None  # package 에 포함된 이미지의 수는 batch_size 의 배수
        self._image_shape = None  # 입력 이미지의 차원
        self._limit_package_size = 0  # package 의 최대 크기 (단위: GBit)
        self._package_size = 0  # package 에 포함된 이미지 수
        self._image_count = 0  # package 들에 저장된 총 이미지 수

    def get_image_count(self):
        return self._image_count

    def clear(self):
        """
            현재 적재된 package 의 메모리를 할당 해제한다.
        """
        if self._package is None:
            return

        del self._package
        self._package = None
        gc.collect()
        self._current_package = -1

    def load_package(self, package_index):
        if self._package_dir is None:
            return

        if self._current_package == package_index:
            return

        self.clear()
        self._package = np.load(os.path.join(self._package_dir, f"package_{package_index}.npy"))
        self._current_package = package_index

    def load_batch(self, batch_index):
        if self._package_dir is None:
            return None

        package_index, start = get_package_index(batch_index, self._batch_size, self._package_size)
        self.load_package(package_index)

        start = start * self._batch_size
        return self._package[start:start+self._batch_size]

    def delete_packages(self):
        """
            임시 경로에 저장된 numpy file 을 제거한다.
        """
        if self._package_dir is None:
            return
        self.clear()

        for i in range(0, self._package_count):
            package_path = os.path.join(self._package_dir, f"package_{i}.npy")
            if os.path.isfile(package_path):
                os.remove(package_path)

        self._package_dir = None
        self._package_count = 0
        self._batch_size = None
        self._image_shape = None
        self._limit_package_size = 0
        self._package_size = 0
        self._image_count = 0

    def create_packages(self, file_list, package_dir, batch_size, image_shape, limit_package_size=4):
        """
            이미지 파일들을 로드하여 package 들을 생성한다.
        """
        package_size = get_package_size(batch_size, image_shape, limit_package_size)
        image_count = len(file_list)
        failed_count = 0
        image_list = []
        package_index = 0
        package_count = 0

        # check package size
        if package_size < 1:
            print(f"Error: too big image shape")
            return 0

        # print information
        print(f"image count per package: {package_size}")

        # delete packages
        self.delete_packages()

        # create packages
        for index in range(0, image_count):
            if not (package_index < package_size):
                # save package
                package_path = os.path.join(package_dir, f"package_{package_count}.npy")
                package = np.array(image_list, dtype=np.uint8)
                np.save(package_path, package)

                del package, image_list
                image_list = []
                package_index = 0
                package_count += 1

            image = cv.imread(file_list[index], cv.IMREAD_COLOR)

            if image is None:
                # failed to loading image
                failed_count += 1
                continue

            if image.shape != image_shape:
                # resize image
                image = cv.resize(image, dsize=(image_shape[0], image_shape[1]), interpolation=cv.INTER_CUBIC)

            image_list.append(image)
            package_index += 1

        # create last package
        package_path = os.path.join(package_dir, f"package_{package_count}.npy")
        package = np.array(image_list, dtype=np.uint8)
        np.save(package_path, package)
        package_count += 1

        # adjust image count
        image_count -= failed_count
        print(f"total package count: {package_count}")

        # set information
        self._package_dir = package_dir
        self._package_count = package_count
        self._batch_size = batch_size
        self._image_shape = image_shape
        self._limit_package_size = limit_package_size
        self._package_size = package_size
        self._image_count = image_count

        return image_count

    def reuse_packages(self, image_count, package_dir, batch_size, image_shape, limit_package_size=4):
        """
            생성되어 있는 package 들을 재사용한다.

            빠른 학습 재시작을 위하여 사용된다.
            package 들이 주어진 정보에 맞게 저장되어 있는지 확인하지 않는다. (사용에 주의)
        """
        self.clear()

        package_size = get_package_size(batch_size, image_shape, limit_package_size)
        package_count = int(np.ceil(image_count / package_size))

        self._package_dir = package_dir
        self._package_count = package_count
        self._batch_size = batch_size
        self._image_shape = image_shape
        self._limit_package_size = limit_package_size
        self._package_size = package_size
        self._image_count = image_count



