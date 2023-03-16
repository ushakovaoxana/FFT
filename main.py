import numpy as np

import time

def ct_fft(x):
    """Compute the FFT of a sequence using the Cooley-Tukey algorithm."""
    n = len(x)
    if n == 1:
        return x
    else:
        # Divide the sequence into even and odd-indexed subsequences
        x_even = ct_fft(x[::2])
        x_odd = ct_fft(x[1::2])
        # Combine the results using twiddle factors
        factors = np.exp(-2j * np.pi * np.arange(n) / n)
        return np.concatenate([x_even + factors[:n//2] * x_odd,
                               x_even + factors[n//2:] * x_odd])

def radix2_fft(x):
    """Compute the FFT of a sequence of length N = 2^k using the Radix-2 FFT algorithm."""
    N = len(x)
    if N == 1:
        return x
    else:
        # Split the sequence into even and odd-indexed subsequences
        x_even = x[::2]
        x_odd = x[1::2]
        # Compute the FFTs of the two subsequences recursively
        X_even = radix2_fft(x_even)
        X_odd = radix2_fft(x_odd)
        # Combine the two FFTs using twiddle factors
        factors = np.exp(-2j * np.pi * np.arange(N) / N)
        X = np.concatenate([X_even + factors[:N//2] * X_odd,
                             X_even + factors[N//2:] * X_odd])
        return X

def bluestein_fft(x):
    N = len(x)
    M = 2 * N - 1
    next_pow2 = int(2**np.ceil(np.log2(M)))
    factor = np.exp(-2j * np.pi * np.arange(N)**2 / M)
    x_padded = np.pad(x, (0, next_pow2 - N))
    factor_padded = np.pad(factor, (0, next_pow2 - N))
    y_padded = np.fft.ifft(np.fft.fft(x_padded) * np.fft.fft(factor_padded))
    return y_padded[:N]

def prime_factor_fft(x):
    N = len(x)
    factors = get_prime_factors(N)
    y = np.copy(x)
    for f in factors:
        for i in range(f):
            block = y[i::f]
            y[i::f] = bluestein_fft(block)
    return y

def get_prime_factors(n):
    factors = []
    p = 2
    while p * p <= n:
        while n % p == 0:
            factors.append(p)
            n //= p
        p += 1
    if n > 1:
        factors.append(n)
    return factors


for N in [2 ** n for n in range(10, 15)]:
    x = np.random.rand(N) + 1j * np.random.rand(N)

    # Time the execution of Bluestein's Algorithm
    start_time = time.time()
    y_bluestein = bluestein_fft(x)
    end_time = time.time()
    bluestein_time = end_time - start_time

    # Time the execution of Prime Factor Algorithm
    start_time = time.time()
    y_prime_factor = prime_factor_fft(x)
    end_time = time.time()
    prime_factor_time = end_time - start_time

    # Time the execution of Radix2 Algorithm
    start_time = time.time()
    y_radix2 = radix2_fft(x)
    end_time = time.time()
    radix2_time = end_time - start_time

    # Time the execution of Cooley-Tukey Algorithm
    start_time = time.time()
    y_ct = ct_fft(x)
    end_time = time.time()
    ct_time = end_time - start_time


    print(f"N = {N}, Cooley-Turkey  FFT time = {ct_time:.6f}")
    print(f"N = {N}, Radix2  FFT time = {radix2_time:.6f}")
    print(f"N = {N}, Bluestein's Algorithm  FFT time = {bluestein_time:.6f}")
    print(f"N = {N}, Prime Factor Algorithm  time = {prime_factor_time:.6f}")


