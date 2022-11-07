import numpy as np
from PIL import Image

from convolutionfilter.conv import Conv


def conv_from_file(
        img_file: str, matrix: np.ndarray, number_of_workers: int, iterations: int, result_file: str = None
) -> None:
    img = np.asarray(Image.open(img_file))
    conv(img, matrix, number_of_workers, iterations, result_file)


def conv(
        img: np.ndarray, matrix: np.ndarray, number_of_workers: int = 1, iterations: int = 1, result_file: str = None
) -> None:
    f = Conv(img, matrix, number_of_workers, iterations)
    f.apply()
    f.save_result(result_file)


MATRIX = {
    "blur1": np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]),
    "blur2": np.array([
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1]
    ]),
    "blur3": np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]),
    "sharpen1": np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]),
    "sharpen2": np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ]),
    "sharpen3": np.array([
        [1, -2, 1],
        [-2, 5, -2],
        [1, -2, 1]
    ])
}
