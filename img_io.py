import os
import numpy as np
import cv2


class Dataset:
    def __init__(self, path):
        self._path = path
        self._batch_size = None
        self._file_paths: list = []

        self.images = []
        self._total_image = 0

    def __str__(self):
        log = f"""Total high image: {len(self._file_paths[0])}\n
                  Total low image {len(self._file_paths[1])}:"""
        return log

    def take_file_paths(self):
        _paths = [os.path.join(self._path, 'high'), os.path.join(self._path, 'low')]
        for p in _paths:
            ls = os.listdir(p)
            temp = [os.path.join(p, f) for f in ls if f.endswith('.jpeg')]
            self._file_paths.append(temp)
        self._total_image = len(self._file_paths[0] * 2)

        self._file_paths = self._file_paths[:30000]

    def load_batch(self, batch):
        if self._batch_size is not None:
            self.images = []
            self.images.clear()
            for high_low in range(2):
                temp_list = []
                if len(self._file_paths[high_low][batch*self._batch_size:]) >= self._batch_size:
                    for b in range(self._batch_size):
                        file = self._file_paths[high_low][batch*self._batch_size + b]
                        temp = cv2.imread(file, cv2.IMREAD_COLOR)
                        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                        temp = temp * 1. / 255
                        temp_list.append(temp)
                else:
                    rang = len(self._file_paths[high_low][:]) - len(self._file_paths[high_low][batch*self._batch_size:])
                    for b in range(rang):
                        file = self._file_paths[high_low][batch*self._batch_size + b]
                        temp = cv2.imread(file, cv2.IMREAD_COLOR)
                        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
                        temp = temp * 1. / 255
                        temp_list.append(temp)
                self.images.append(temp_list)
            self.images = np.array(self.images)
            return np.array(self.images[0]), np.array(self.images[1])

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self._batch_size = batch_size

    @property
    def total_image(self):
        return self._total_image
