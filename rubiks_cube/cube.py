import numpy as np
import random
import copy
from sympy.utilities.iterables import multiset_permutations
from PIL import Image

# import torch - for later Tensor support

INT_COLOR_MAP = {
    0: "w",
    1: "r",
    2: "g",
    3: "b",
    4: "y",
    5: "o"}

COLOR_INT_MAP = {v: k for k, v in INT_COLOR_MAP.items()}
COLOR_RGB_MAP = {
    'w': [255, 255, 255],
    'r': [255, 0, 0],
    'g': [0, 255, 0],
    'b': [0, 0, 255],
    'y': [255, 255, 0],
    'o': [255, 168, 0]
}
INT_RGB_MAP = {COLOR_INT_MAP[k]: v for k, v in COLOR_RGB_MAP.items()}


class Cube:
    """
    #TODO: Is a 'Face' class useful here?

    """

    def __init__(self, n: int = None, debug: bool = False, init_state: np.ndarray = None):

        if n is None and init_state is None:
            raise RuntimeError("At least n or init_state needs to be specified.")

        if init_state is not None:
            # if (
            #         len(init_state.shape) == 3 or
            #         init_state.shape[0] != 6 or
            #         init_state.shape[1] != init_state.shape[2] or
            #         (n is not None and n != init_state.shape[1])):
            #     raise ValueError(f"init_state must be of shape (6,n,n) but found {init_state.shape}")
            self._sides = init_state
            self._n = init_state.shape[1]

        else:
            self._n = n
            if not debug:
                self._sides = np.ones((6, self.n, self.n), dtype=int)
                for side, v in zip(self._sides, INT_COLOR_MAP.keys()):
                    side *= v
            else:
                self._sides = [i for i in range(6 * n * n)]
            self._sides = np.reshape(self._sides, (6, n, n))

        self.init_state = copy.deepcopy(self._sides)

    def __str__(self):
        """
        Dumb implementation, but works for now.
        """
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

    def to_img(self, block_size=64, border_size=16):
        """
        Dumb implementation - raster line by line like the str method.

        """
        img = []
        # Top n lines
        for _ in range(border_size):
            line = []
            for _ in range(border_size):
                line.append([0, 0, 0])
            for _ in range((border_size + block_size) * 4 * self.n):
                line.append([0, 0, 0])
            img.append(line)

        for i in range(self.n):
            for _ in range(block_size):
                line = []
                for _ in range(border_size):
                    line.append([0, 0, 0])

                for _ in range(self.n):
                    for _ in range(block_size + border_size):
                        line.append([0, 0, 0])
                for j in range(self.n):
                    for _ in range(block_size):
                        line.append(INT_RGB_MAP[self.sides[0, i, j]])
                    for _ in range(border_size):
                        line.append([0, 0, 0])
                for _ in range(2 * self.n):
                    for _ in range(block_size + border_size):
                        line.append([0, 0, 0])
                img.append(line)

            for _ in range(border_size):
                line = []
                for _ in range(border_size):
                    line.append([0, 0, 0])
                for _ in range((border_size + block_size) * 4 * self.n):
                    line.append([0, 0, 0])
                img.append(line)

        for i in range(self.n):
            for _ in range(block_size):
                line = []
                for _ in range(border_size):
                    line.append([0, 0, 0])

                for side in range(1, 5):
                    for j in range(self.n):
                        for _ in range(block_size):
                            line.append(INT_RGB_MAP[self.sides[side, i, j]])
                        for _ in range(border_size):
                            line.append([0, 0, 0])
                img.append(line)

            for _ in range(border_size):
                line = []
                for _ in range(border_size):
                    line.append([0, 0, 0])

                for _ in range((border_size + block_size) * 4 * self.n):
                    line.append([0, 0, 0])
                img.append(line)

        for i in range(self.n):
            for _ in range(block_size):
                line = []
                for _ in range(border_size):
                    line.append([0, 0, 0])

                for _ in range(self.n):
                    for _ in range(block_size + border_size):
                        line.append([0, 0, 0])
                for j in range(self.n):
                    for _ in range(block_size):
                        line.append(INT_RGB_MAP[self.sides[5, i, j]])
                    for _ in range(border_size):
                        line.append([0, 0, 0])
                for _ in range(2 * self.n):
                    for _ in range(block_size + border_size):
                        line.append([0, 0, 0])
                img.append(line)

            for _ in range(border_size):
                line = []
                for _ in range(border_size):
                    line.append([0, 0, 0])
                for _ in range((border_size + block_size) * 4 * self.n):
                    line.append([0, 0, 0])
                img.append(line)

        return np.array(img, dtype=np.uint8)

    def make_random_moves(self, N):
        for _ in range(N):
            axis = random.randint(0, 2)
            direction = random.choice([-1, 1])
            idx = random.randint(0, self.n - 1)

            self.move(axis, idx, direction)

    def move(self, axis, idx, direction):
        """
        Probably an easier way exists by first rotating to a common representation
        based on Axis.

        But, do the dumb way that works first and double-check with real cube.

        Then can use this as a test for a smarter implementation.

        """
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
            self.sides[4, :, :] = np.rot90(self.sides[4, :, :], 2)
            self.sides[:, :, idx] = self.sides[rot_indexer, :, idx]
            self.sides[4, :, :] = np.rot90(self.sides[4, :, :], -2)
            if idx == self.n - 1:
                self.sides[3, :, :] = np.rot90(self.sides[3, :, :], direction)
            if idx == 0:
                self.sides[1, :, :] = np.rot90(self.sides[1, :, :], -direction)

    def reset(self):
        self._sides = copy.deepcopy(self.init_state)


def permutate_state_colors(cube_state):
    """
    Data augmentation ?
    Every color can be swapped with every other color

    There's probably a better way to handle 'color symmetry'
    but brute forcing it this way might work initially?

    Since each color can be substituted for any other color,
    for a given cube configuration, we can populate it 6! == 720
    different ways.

    This function returns all 720 combinations possible for a
    given configuration
    """
    all_cube_states = np.zeros((720,) + cube_state.shape)
    source_map = np.array([i for i in range(6)])
    for i, target_map in enumerate(multiset_permutations(source_map)):
        all_cube_states[i, :, :, :] = \
            np.select([cube_state == a for a in source_map], target_map)

    return all_cube_states


def rotate_right(cube_state):
    """
    Data augmentation ?

    Rotating the reference frame is valid.
    There might be a better way to handle rotational 'symmetry'

    Can also then rotate the cube 90, 180 or 270 degrees.

    """
    rot_indexer = np.array([1, 5, 2, 0, 4, 3])
    return np.rot90(cube_state[rot_indexer], axes=(1, 2), k=-1)


def flip_cube_left(cube_state):
    """
    Data augmentation?

    Flipping the cube such that a new face becomes '0'
    gives rise to new valid states

    """
    raise NotImplementedError


def flip_cube_down(cube_state):
    raise NotImplementedError
