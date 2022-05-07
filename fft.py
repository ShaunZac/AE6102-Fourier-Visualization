import time
import numba
import numpy as np


@numba.jit
def ilog2(n):
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
    levels = ilog2(n)
    e_arr = np.zeros_like(arr)
    coeff = -2j * np.pi / n
    for i in range(n):
        e_arr[i] = np.exp(coeff * i)
    result = np.empty_like(arr)
    for i in range(n):
        result[i] = arr[bitrev(i, levels)]

    size = 2
    while size <= n:
        half_size = size // 2
        step = n // size
        for i in range(0, n, size):
            k = 0
            for j in range(i, i + half_size):
                temp = result[j + half_size] * e_arr[k]
                result[j + half_size] = result[j] - temp
                result[j] += temp
                k += step
        size *= 2
    return result


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


if __name__ == "__main__":
    fft2d(np.ones((256, 256)))
    array = np.ones((256, 256))
    start = time.time()
    fft2d(array)
    print(f"Time taken {time.time() - start} s")