import numpy as np
import copy
from PIL import Image

# import torch - for later Tensor support

INT_COLOR_MAP = {0: "w", 1: "r", 2: "g", 3: "b", 4: "y", 5: "o"}
COLOR_INT_MAP = {v: k for k, v in INT_COLOR_MAP.items()}
COLOR_RGB_MAP = {
    'w': [255, 255, 255],
    'r': [255, 0, 0],
    'g': [0, 255, 0],
    'b': [0, 0, 255],
    'y': [255, 255, 0],
    'o': [255, 215, 0]
}
INT_RGB_MAP = {COLOR_INT_MAP[k]: v for k, v in COLOR_RGB_MAP.items()}


class Cube:
    def __init__(self, n=3, debug=False):
        self._n = n
        if not debug:
            self._sides = np.ones((6, self.n, self.n), dtype=int)
            for side, v in zip(self._sides, INT_COLOR_MAP.keys()):
                side *= v
        else:
            self._sides = [i for i in range(6 * n * n)]
            self._sides = np.reshape(self._sides, (6, n, n))

    def __str__(self):
        return_str = ""
        filler = "   "
        for i in range(self.n):
            return_str += filler * self.n
            for j in range(self.n):
                return_str += f" {self.sides[0, i, j]} "
            return_str += filler * self.n * 2
            return_str += "\n"

        for i in range(self.n):
            for j in range(self.n):
                return_str += f" {self.sides[1, i, j]} "
            for j in range(self.n):
                return_str += f" {self.sides[2, i, j]} "
            for j in range(self.n):
                return_str += f" {self.sides[3, i, j]} "
            for j in range(self.n):
                return_str += f" {self.sides[4, i, j]} "
            return_str += "\n"

        for i in range(self.n):
            return_str += filler * self.n
            for j in range(self.n):
                return_str += f" {self.sides[5, i, j]} "
            return_str += filler * self.n * 2
            return_str += "\n"

        return return_str

    def __repr__(self):
        return str(self)

    @property
    def n(self):
        return self._n

    @property
    def sides(self):
        return self._sides

    def to_img(self, block_size=64):
        raise NotImplementedError()

    def move(self, axis, idx, direction):
        assert idx < self.n
        assert axis in [0, 1, 2]
        assert direction in [-1, 1]

        axis_face_map = {0: 0, 1: 2, 2: 3}

        # Hard coding these for now
        # Do it smarter when I'm more awake
        if axis == 0:
            if direction == -1:
                rot_indexer = [0, 2, 3, 4, 1, 5]
            elif direction == 1:
                rot_indexer = [0, 4, 1, 2, 3, 5]

        if axis == 1:
            if direction == -1:
                rot_indexer = [1, 5, 2, 0, 4, 3]
            elif direction == 1:
                rot_indexer = [3, 0, 2, 5, 4, 1]

        if axis == 2:
            if direction == -1:
                rot_indexer = [2, 1, 5, 3, 0, 4]
            elif direction == 1:
                rot_indexer = [4, 1, 0, 3, 5, 2]

        if axis == 0:
            self.sides[:, idx, :] = self.sides[rot_indexer, idx, :]
            if idx == 0:
                self.sides[0, :, :] = np.rot90(self.sides[0, :, :], direction)
            if idx == self.n - 1:
                self.sides[5, :, :] = np.rot90(self.sides[5, :, :], -direction)

        if axis == 1:
            idx = self.n - idx - 1
            self.sides[1, :, :] = np.rot90(self.sides[1, :, :], -1)
            self.sides[3, :, :] = np.rot90(self.sides[3, :, :], 1)
            self.sides[5, :, :] = np.rot90(self.sides[5, :, :], 2)
            self.sides[:, idx, :] = self.sides[rot_indexer, idx, :]
            self.sides[1, :, :] = np.rot90(self.sides[1, :, :], 1)
            self.sides[3, :, :] = np.rot90(self.sides[3, :, :], -1)
            self.sides[5, :, :] = np.rot90(self.sides[5, :, :], -2)

            if idx == self.n - 1:
                self.sides[2, :, :] = np.rot90(self.sides[2, :, :], direction)
            if idx == 0:
                self.sides[4, :, :] = np.rot90(self.sides[4, :, :], -direction)

        if axis == 2:
            idx = self.n - idx - 1
            self.sides[:, :, idx] = self.sides[rot_indexer, :, idx]
            if idx == self.n - 1:
                self.sides[3, :, :] = np.rot90(self.sides[2, :, :], direction)
            if idx == 0:
                self.sides[1, :, :] = np.rot90(self.sides[4, :, :], -direction)
