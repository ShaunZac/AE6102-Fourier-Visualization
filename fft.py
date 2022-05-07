import time
import numba
import numpy as np


def bitrev(x):
    """
    Funtion for bit reversal
    Note that you are working with 8 point fft
    So you only need 3 bits to be reversed
    """
    units_place_bit = x % 2
    tens_place_bit = (x // 2) % 2
    hund_place_bit = ((x // 2) // 2) % 2
    return (4 * units_place_bit) + (2 * tens_place_bit) + hund_place_bit


def complex_multiply(x0, x1):
    """
    Multiplication of 2 complex numbers
    Calculate the real and imag part separately and return
    the complex number
    """
    real = (np.real(x0) * np.real(x1)) - (np.imag(x0) * np.imag(x1))
    imag = (np.real(x0) * np.imag(x1)) + (np.imag(x0) * np.real(x1))
    return real + (1j * imag)


def butterfly(x0, x1):
    """
    Butterfly operation
    Takes in 2 numbers x0 and x1 and returns
    2 fft values y0 and y1
    """
    y0 = x0 + x1
    y1 = x0 - x1
    return y0, y1


def fft_1d(x):
    """
    1D implementation of FFT algorithm
    """
    N = len(x)
    y = np.zeros(N, dtype=np.complex128)
    if N == 1:
        y[0] = x[0]
    else:
        N_by_2 = int(N/2)
        x1 = np.zeros(N_by_2, dtype=np.complex128)
        x2 = np.zeros(N_by_2, dtype=np.complex128)
        for i in range(N):
            if i % 2 == 0:
                x1[i // 2] = x[i]
            else:
                x2[i // 2] = x[i]
        y1 = fft_1d(x1)
        y2 = fft_1d(x2)
        for k in range(N_by_2):
            twiddle = (np.cos((2 * np.pi * k) / N)) - (1j * np.sin((2 * np.pi * k) / N))
            y2[k] = complex_multiply(twiddle, y2[k])
            y[k], y[k + N_by_2] = butterfly(y1[k], y2[k])
    return y


def fft_rows(arr, fft_arr, xsize):
    """
    Takes FFT of all rows
    """
    for i in range(xsize):
        fft_arr[i] = fft_1d(arr[i])
    return


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
