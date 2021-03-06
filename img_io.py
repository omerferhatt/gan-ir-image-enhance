import numpy as np
import os
import cv2
import random


class Dataset:
    def __init__(self, path, shuffle=1):
        self._path: str = path
        self._batch_size = None
        self.file_paths, self._total_image = self.take_file_paths()

        self.shuffle: int = shuffle
        if self.shuffle != 0:
            self.shuffle_dataset(shuffle)

        self.images: list = []
        self.sample_images = self.load_sample_images()

    def __str__(self):
        log = f"""Total high image: {self._total_image}, Total low image {self._total_image}:"""
        return log

    def take_file_paths(self):
        _paths = [os.path.join(self._path, 'high'), os.path.join(self._path, 'low')]
        image_paths = []
        for p in _paths:
            ls = os.listdir(p)
            temp = [os.path.join(p, f) for f in ls if f.endswith('.jpeg')]
            temp.sort()
            image_paths.append(temp)

        total_image = len(image_paths[0])
        return image_paths, total_image

    def shuffle_dataset(self, seed):
        random.seed(seed)
        random.shuffle(self.file_paths[0])
        random.seed(seed)
        random.shuffle(self.file_paths[1])

    def load_batch(self, batch):
        if self._batch_size is not None:
            self.images = []
            self.images.clear()

            def read_image(file: str) -> np.ndarray:
                temp = cv2.imread(file, cv2.IMREAD_COLOR)
                temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                return temp / 127.5 - 1

            for high_low in range(2):
                temp_list_ = []
                if len(self.file_paths[high_low][batch * self._batch_size:]) >= self._batch_size:
                    for b in range(self._batch_size):
                        file_ = self.file_paths[high_low][batch * self._batch_size + b]
                        temp_ = read_image(file_)
                        temp_list_.append(temp_)
                else:
                    for b in range(len(self.file_paths[high_low][batch * self._batch_size:])):
                        file_ = self.file_paths[high_low][batch * self._batch_size + b]
                        temp_ = read_image(file_)
                        temp_list_.append(temp_)
                self.images.append(temp_list_)
            self.images = np.array(self.images)
            return np.array(self.images[0]), np.array(self.images[1])

    @staticmethod
    def load_sample_images():
        path = '/home/ferhat/PycharmProjects/gan-ir-image-enhance/sample_input'
        ls = os.listdir(path)
        full_dir = [os.path.join(path, p) for p in ls]
        temp = []
        for sample in full_dir:
            img = cv2.imread(sample, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            temp.append(img / 127.5 - 1)

        return np.array(temp)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def total_image(self):
        return self._total_image
