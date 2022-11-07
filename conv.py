import sys
import time
from typing import Callable

from convolutionfilter.api import conv_from_file, MATRIX


def current_time() -> int:
    return int(time.time() * 1000)


def measure_time(start: int) -> int:
    return current_time() - start


def run_timed(task: Callable[[], None]):
    start = current_time()
    task()
    elapsed = measure_time(start)
    return elapsed


def run_benchmark(img_file: str, max_workers: int, max_iterations: int):
    def run(w: int, i: int): conv_from_file(img_file, MATRIX['blur1'], w, i, f'result_{w}_{i}')
    WORKERS_STEP = 2
    ITERATIONS_STEP = 2

    result = run_timed(lambda: run(1, 1))
    print(f'{1} {1} {result}')
    for workers in range(2, max_workers+1, WORKERS_STEP):
        result = run_timed(lambda: run(workers, 1))
        print(f'{workers} {1} {result}')
        for iterations in range(2, max_iterations+1, ITERATIONS_STEP):
            result = run_timed(lambda: run(workers, iterations))
            print(f'{workers} {iterations} {result}')
            time.sleep(2)


def app(img_file: str, matrix: str, workers: int, iterations: int, timed: str):
    def run(): conv_from_file(img_file, MATRIX[matrix], workers, iterations)

    if timed is not None:
        elapsed = run_timed(run)
        print(elapsed)
    else:
        run()


if __name__ == '__main__':
    args = sys.argv[1:]
    if args[0] == 'bench':
        run_benchmark(args[1], int(args[2]), int(args[3]))
    else:
        app(args[0], args[1], int(args[2]), int(args[3]), args[4] if len(args) == 5 else None)
