#!/usr/bin/env python3
"""Find optimal degree for log2(1+t) on [0,1] in 63-bit fixed point."""

import mpmath as mp
mp.mp.dps = 80

def chebyshev_nodes(n, a, b):
    return [(b + a) / 2 + (b - a) / 2 * mp.cos(mp.pi * (2*k + 1) / (2*n)) for k in range(n)]

def chebyshev_coefficients(f, n, a, b):
    nodes = chebyshev_nodes(n, a, b)
    fvals = [f(x) for x in nodes]
    coeffs = []
    for j in range(n):
        s = sum(fvals[k] * mp.cos(j * mp.acos(mp.cos(mp.pi * (2*k + 1) / (2*n)))) for k in range(n))
        c = 2 * s / n
        if j == 0: c /= 2
        coeffs.append(c)
    return coeffs

def chebyshev_to_monomial(coeffs, a, b):
    n = len(coeffs)
    mono = [mp.mpf(0)] * n
    T_prev = [mp.mpf(0)] * n; T_prev[0] = 1
    T_curr = [mp.mpf(0)] * n
    if n > 1: T_curr[1] = 1
    for i in range(n): mono[i] += coeffs[0] * T_prev[i]
    if n > 1:
        for i in range(n): mono[i] += coeffs[1] * T_curr[i]
    for k in range(2, n):
        T_next = [mp.mpf(0)] * n
        for i in range(n):
            if i > 0: T_next[i] += 2 * T_curr[i-1]
            T_next[i] -= T_prev[i]
        for i in range(n): mono[i] += coeffs[k] * T_next[i]
        T_prev, T_curr = T_curr, T_next
    alpha = mp.mpf(2) / (b - a); beta = -(a + b) / (b - a)
    result = [mp.mpf(0)] * n
    for k in range(n):
        binom = [mp.mpf(0)] * n; binom[0] = 1
        for _ in range(k):
            nb = [mp.mpf(0)] * n
            for i in range(n):
                nb[i] += beta * binom[i]
                if i > 0: nb[i] += alpha * binom[i-1]
            binom = nb
        for i in range(n): result[i] += mono[k] * binom[i]
    return result

SCALE = mp.power(2, 63)

f = lambda t: mp.log(1 + t, 2)

print("Approach 2: log2(1+t) on [0, 1]")
for n in range(13, 30):
    mono = chebyshev_to_monomial(chebyshev_coefficients(f, n, 0, 1), 0, 1)
    horner = list(reversed(mono))
    coef_q = [int(mp.nint(c * SCALE)) for c in horner]

    max_err = mp.mpf(0)
    for i in range(10001):
        t = mp.mpf(i) / 10000
        t_q = int(mp.nint(t * SCALE))
        y = 0
        for c in coef_q:
            y = int(mp.mpf(y) * mp.mpf(t_q) / SCALE) + c
        result = mp.mpf(y) / SCALE
        err = abs(result - f(t))
        if err > max_err: max_err = err

    bits = -float(mp.log(max_err, 2)) if max_err > 0 else 999
    print(f"  n={n} (deg {n-1}): {bits:.2f} bits  err={float(max_err):.3e}  {'PASS' if bits >= 63 else ''}")
    if bits >= 63:
        print(f"\n  ==> Use degree {n-1}")
        # Output coefficients
        for i, c in enumerate(horner):
            print(f"    horner[{i}] = {float(c):.25e}")
        # Output quantized
        print(f"\n  Quantized (integers):")
        for i, cq in enumerate(coef_q):
            sign = "-" if cq < 0 else ""
            print(f"    {sign}{abs(cq)}")
        break
