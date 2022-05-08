import time
import numba
import numpy as np
import matplotlib.pyplot as plt


@numba.jit
def ilog2(n):
    """
    Integer value of inverse log 2
    """
    result = 0
    while n:
        n = n >> 1
        result += 1
    return result - 1


@numba.njit
def bitrev(val, width):
    """
    Function for bit reversal
    """
    result = 0
    for _ in range(width):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


@numba.njit
def fft_1d(arr):
    """
    1D FFT, works when array size is power of 2
    """
    arr = np.asarray(arr, dtype=np.complex128)
    n = len(arr)
    bits = ilog2(n)
    exp_arr = np.zeros_like(arr)
    coeff = -2j * np.pi / n
    for i in range(n):
        exp_arr[i] = np.exp(coeff * i)

    fft_result = np.zeros_like(arr)
    for i in range(n):
        fft_result[i] = arr[bitrev(i, bits)]

    size = 2
    while size <= n:
        half_size = size // 2
        step = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + half_size):
                temp_val = fft_result[j + half_size] * exp_arr[k]
                fft_result[j + half_size] = fft_result[j] - temp_val
                fft_result[j] += temp_val
                k = k + step
        size *= 2
    return fft_result


@numba.njit
def fft_rows(arr, fft_arr, xsize):
    """
    Takes FFT of all rows
    """
    for i in range(xsize):
        fft_arr[i] = fft_1d(arr[i])
    return


@numba.njit
def fft_cols(fft_arr, ysize):
    """
    Takes FFT of all columns
    """
    for i in range(ysize):
        fft_arr[:, i] = fft_1d(fft_arr[:, i])
    return


def fft2d(arr):
    """
    2D fourier transform using 1D FFT
    """
    arr = np.array(arr)
    fft_arr = np.zeros_like(arr, dtype=np.complex128)
    fft_rows(arr, fft_arr, fft_arr.shape[0])
    fft_cols(fft_arr, fft_arr.shape[1])
    return fft_arr


def avg_time(arr, func, num_iters=100):
    times = np.zeros(num_iters)
    for i in range(num_iters):
        start = time.time()
        func(arr)
        end = time.time()
        times[i] = end - start
    return np.mean(times)


def fn_performance(func, num_iters=100):
    np.random.seed(0)
    arr_16 = np.random.random((16, 16))
    arr_32 = np.random.random((32, 32))
    arr_64 = np.random.random((64, 64))
    arr_128 = np.random.random((128, 128))
    arr_256 = np.random.random((256, 256))
    arr_512 = np.random.random((512, 512))
    arr_1024 = np.random.random((1024, 1024))

    times = [
        avg_time(arr_16, func, num_iters),
        avg_time(arr_32, func, num_iters),
        avg_time(arr_64, func, num_iters),
        avg_time(arr_128, func, num_iters),
        avg_time(arr_256, func, num_iters),
        avg_time(arr_512, func, num_iters),
        avg_time(arr_1024, func, num_iters)
    ]
    return times


def plot_times(times, legend=["FFT times"]):
    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 16})
    for time_plt in times:
        plt.plot([2**i for i in range(4, 11)], time_plt)
    plt.title("Time vs FFT size")
    plt.legend(legend)
    plt.xlabel("Size of 2D FFT Transform")
    plt.ylabel("Time taken (s)")
    plt.show()


if __name__ == "__main__":
    fft2d(np.ones((4, 4)))
    times_numba_fft = np.array(fn_performance(fft2d))
    times_np_fft = np.array(fn_performance(np.fft.fft))
    plot_times([times_numba_fft, times_np_fft], ["Numba FFT", "Numpy FFT"])
