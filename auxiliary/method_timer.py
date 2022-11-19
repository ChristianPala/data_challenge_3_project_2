# Auxiliary library to time the execution of a function or a method, inherited from
# the project 1 source code from Christian Berchtold and Christian Pala.
# Libraries:
from functools import wraps
import time
from typing import Callable, Any


# Functions:
def measure_time(func: Callable) -> Callable:
    """
    Decorator to measure the execution time of a function or a method.
    @param func: The function or method to be timed.
    :return: The function or method wrapped with the timer.
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.3f} seconds to complete')
        return result
    return timeit_wrapper

