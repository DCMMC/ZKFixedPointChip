#!/usr/bin/env python3
"""Generate polynomial for log2(1+t) on [0, 1] with high-precision quantization."""

import mpmath as mp
import math

mp.mp.dps = 80

def chebyshev_nodes(n, a, b):
    nodes = []
    for k in range(n):
        t = mp.cos(mp.pi * (2*k + 1) / (2*n))
        x = (b + a) / 2 + (b - a) / 2 * t
        nodes.append(x)
    return nodes

def chebyshev_coefficients(f, n, a, b):
    nodes = chebyshev_nodes(n, a, b)
    fvals = [f(x) for x in nodes]
    coeffs = []
    for j in range(n):
        s = mp.mpf(0)
        for k in range(n):
            t = mp.cos(mp.pi * (2*k + 1) / (2*n))
            Tj = mp.cos(j * mp.acos(t))
            s += fvals[k] * Tj
        c = 2 * s / n
        if j == 0:
            c /= 2
        coeffs.append(c)
    return coeffs

def chebyshev_to_monomial(coeffs, a, b):
    n = len(coeffs)
    mono = [mp.mpf(0)] * n
    T_prev = [mp.mpf(0)] * n
    T_prev[0] = mp.mpf(1)
    T_curr = [mp.mpf(0)] * n
    if n > 1:
        T_curr[1] = mp.mpf(1)
    for i in range(n):
        mono[i] += coeffs[0] * T_prev[i]
    if n > 1:
        for i in range(n):
            mono[i] += coeffs[1] * T_curr[i]
    for k in range(2, n):
        T_next = [mp.mpf(0)] * n
        for i in range(n):
            if i > 0:
                T_next[i] += 2 * T_curr[i-1]
            T_next[i] -= T_prev[i]
        for i in range(n):
            mono[i] += coeffs[k] * T_next[i]
        T_prev = T_curr
        T_curr = T_next
    alpha = mp.mpf(2) / (b - a)
    beta = -(a + b) / (b - a)
    result = [mp.mpf(0)] * n
    for k in range(n):
        binom = [mp.mpf(0)] * n
        binom[0] = mp.mpf(1)
        for _ in range(k):
            new_binom = [mp.mpf(0)] * n
            for i in range(n):
                new_binom[i] += beta * binom[i]
                if i > 0:
                    new_binom[i] += alpha * binom[i-1]
            binom = new_binom
        for i in range(n):
            result[i] += mono[k] * binom[i]
    return result

def eval_poly(coeffs, x):
    s = mp.mpf(0)
    for i in reversed(range(len(coeffs))):
        s = s * x + coeffs[i]
    return s

f = lambda t: mp.log(1 + t, 2)
a, b = mp.mpf(0), mp.mpf(1)

PRECISION_BITS = 63
SCALE = mp.power(2, PRECISION_BITS)

# For the approach where we keep the polynomial on [2,4] directly,
# but quantize coefficients with full precision:
# log2(x) on [2,4], x in fixed point
f_direct = lambda x: mp.log(x, 2)
a_d, b_d = mp.mpf(2), mp.mpf(4)

for approach in ["direct_[2,4]", "substitution_[0,1]"]:
    if approach == "direct_[2,4]":
        func, aa, bb = f_direct, a_d, b_d
        print("=== Approach 1: Direct polynomial on [2, 4] ===")
    else:
        func, aa, bb = f, a, b
        print("\n=== Approach 2: log2(1+t) on [0, 1], t = x/2 - 1 ===")

    for n in range(13, 30):
        cheb_coeffs = chebyshev_coefficients(func, n, aa, bb)
        mono_coeffs = chebyshev_to_monomial(cheb_coeffs, aa, bb)

        # Horner: highest degree first
        horner = list(reversed(mono_coeffs))

        # HIGH-PRECISION quantization (bypass f64)
        def quantize_hp(c):
            """Quantize using mpmath, not f64."""
            return int(mp.nint(c * SCALE))

        coef_q = [quantize_hp(c) for c in horner]

        def qmul_sim(a_val, b_val):
            return int(mp.mpf(a_val) * mp.mpf(b_val) / SCALE)

        max_err_fp = mp.mpf(0)
        worst_x = mp.mpf(0)
        for i in range(10001):
            t = aa + (bb - aa) * mp.mpf(i) / 10000
            t_q = int(mp.nint(t * SCALE))
            y = 0
            for c in coef_q:
                y = qmul_sim(y, t_q) + c
            result = mp.mpf(y) / SCALE
            exact = func(t)
            err = abs(result - exact)
            if err > max_err_fp:
                max_err_fp = err
                worst_x = t

        bits_fp = -float(mp.log(max_err_fp, 2)) if max_err_fp > 0 else 999
        print(f"  n={n} (degree {n-1}):")
        print(f"    Fixed-point max error: {float(max_err_fp):.6e}")
        print(f"    Precision bits: {bits_fp:.2f}")
        print(f"    Worst at: {float(worst_x):.6f}")
        print(f"    {'PASS' if bits_fp >= 63 else 'FAIL'}")

        if bits_fp >= 63:
            # Output quantized coefficients as integers for Rust
            print(f"\n    Quantized coefficients (integers, for direct use as field elements):")
            for i, cq in enumerate(coef_q):
                sign = "-" if cq < 0 else ""
                print(f"      coef_q[{i}] = {sign}{abs(cq)}")
