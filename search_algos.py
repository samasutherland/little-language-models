import math
import functools

def fibonacci_search(func, func_args=(), func_kwargs=None, lower_bound=1, upper_bound=32):
    if func_kwargs is None:
        func_kwargs = {}
    @functools.lru_cache
    def func_(probe):
        print(f"evaluating {probe}")
        return func(probe, *func_args, **func_kwargs)


    fib_nums = [1,1]
    while fib_nums[-1] <= (upper_bound - lower_bound):
        fib_nums.append(fib_nums[-1] + fib_nums[-2])

    fib_nums = fib_nums[::-1][1:]
    fib_index = 0

    # Initial pass uses two evaluations
    probe_upper = lower_bound + fib_nums[fib_index] - 1
    probe_lower = lower_bound + fib_nums[fib_index + 1] - 1

    result_upper = func_(probe_upper)
    result_lower = func_(probe_lower)


    if result_upper < result_lower:
        lower_bound = probe_lower + 1
        probe_lower = probe_upper
        result_lower = result_upper
        flag = "UPPER"
    else:
        upper_bound = probe_upper - 1
        probe_upper = probe_lower
        result_upper = result_lower
        flag = "LOWER"

    fib_index += 1

    while upper_bound - lower_bound > 1:
        if flag == "UPPER":
            probe_upper = lower_bound + fib_nums[fib_index] - 1
            result_upper = func_(probe_upper)

        elif flag == "LOWER":
            probe_lower = lower_bound + fib_nums[fib_index + 1] - 1
            result_lower = func_(probe_lower)

        if result_upper < result_lower:
            lower_bound = probe_lower + 1
            probe_lower = probe_upper
            result_lower = result_upper
            flag = "UPPER"
        else:
            upper_bound = probe_upper - 1
            probe_upper = probe_lower
            result_upper = result_lower
            flag = "LOWER"

        fib_index += 1

    result_upper = func_(upper_bound)
    result_lower = func_(lower_bound)
    if result_upper < result_lower:
        return upper_bound
    else:
        return lower_bound

def binary_search(func, func_args=(), func_kwargs=None, lower_bound=1, upper_bound=32):