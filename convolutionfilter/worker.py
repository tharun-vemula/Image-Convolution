from multiprocessing import Process, Lock
from multiprocessing.managers import BaseManager

import numpy as np


class _WorkerResult:

    def __init__(self, height: int, width: int) -> None:
        self._data = np.empty((height, width, 3), dtype=np.uint8)

    def set_row(self, y: int, value: np.ndarray):
        self._data[y] = value[1:-1]

    def copy_from(self, source: np.ndarray) -> None:
        for i in range(1, len(source)-1):
            self.set_row(i-1, source[i])

    def get(self): return self._data


class _WorkerRow:

    def __init__(self, width: int) -> None:
        self._data = np.empty((width, 3), dtype=np.uint8)

    def set(self, value: np.ndarray):
        self._data = value

    def get(self): return self._data


class _WorkersManager(BaseManager):

    def __init__(self) -> None:
        super().__init__()
        self.start()


_WorkersManager.register('result', _WorkerResult)
_WorkersManager.register('row', _WorkerRow)


class _ConvWorker(Process):

    def __init__(
            self, n: int, runs: int,
            data: np.ndarray, matrix: np.ndarray, result: _WorkerResult,
            first_row: _WorkerRow, last_row: _WorkerRow,
            first_row_lock: Lock, last_row_lock: Lock,
            top_border: _WorkerRow, bottom_border: _WorkerRow,
            top_border_lock: Lock, bottom_border_lock: Lock
    ) -> None:
        super().__init__(name=f'worker {n}')
        self._runs = runs
        self._data = data
        self._matrix = matrix
        self.result = result

        self._height = len(data)
        width = len(data[0])
        self._height_range = range(1, self._height)
        self._width_range = range(1, width + 1)
        self._current_iter = np.pad(self._data, ((1, 1), (1, 1), (0, 0)), 'edge')
        self._next_iter = np.pad(self._data, ((1, 1), (1, 1), (0, 0)), 'edge')
        self._matrix_sum = self._matrix.sum()

        self.first_row = first_row
        self.last_row = last_row
        self.first_row_lock = first_row_lock
        self.last_row_lock = last_row_lock

        self.top_border = top_border
        self.bottom_border = bottom_border
        self.top_border_lock = top_border_lock
        self.bottom_border_lock = bottom_border_lock

        if first_row is not None:
            self.first_row.set(self._current_iter[0])
        self.last_row.set(self._current_iter[self._height])

    def run(self) -> None:
        self._process_iterations()

    def _process_iterations(self):
        for _ in range(0, self._runs):
            self._process_next_iteration()
        self.result.copy_from(self._current_iter)

    def _process_next_iteration(self):
        self._process_rows()
        self._switch_iterations()

    def _switch_iterations(self):
        t = self._current_iter
        self._current_iter = self._next_iter
        self._next_iter = t

    def _process_rows(self) -> None:
        self._process_row(1, self._current_iter[0:3])
        self._synchronize_top()

        for i in self._height_range:
            self._process_row(i, self._current_iter[i - 1:i + 2])

        self._process_row(self._height, self._current_iter[self._height - 1:])
        self._synchronize_bottom()

    def _synchronize_top(self):
        if self.top_border is not None:
            with self.top_border_lock:
                self._current_iter[0] = self.top_border.get()
            with self.first_row_lock:
                self.first_row.set(self._current_iter[1])

    def _synchronize_bottom(self):
        if self.bottom_border is not None:
            with self.bottom_border_lock:
                self._current_iter[self._height + 1] = self.bottom_border.get()
            with self.last_row_lock:
                self.last_row.set(self._current_iter[self._height])

    def _process_row(self, i: int, row: np.ndarray) -> None:
        for j in self._width_range:
            self._next_iter[i][j] = self._process_pixel(row[0:3, j - 1:j + 2])

    def _process_pixel(self, pixel: np.ndarray) -> np.ndarray:
        return np.array([self._calculate_pixel(pixel[0:3, 0:3, i]) for i in (0, 1, 2)], dtype=np.uint8)

    def _calculate_pixel(self, value: np.ndarray) -> int:
        return (value * self._matrix).sum() / self._matrix_sum
