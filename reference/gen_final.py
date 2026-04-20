#!/usr/bin/env python3
"""Output final polynomial coefficients for log2(1+t) on [0,1], degree 22."""
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

# Two options: keep on [2,4] with degree 24 (same as before, good polynomial error)
# or use [0,1] with variable substitution

# For [2,4] direct, we need the circuit to handle it as-is (no code change needed)
# For [0,1] we need circuit changes

# Let's output BOTH and let the user decide

print("=" * 70)
print("Option A: Direct polynomial on [2, 4] (NO circuit change needed)")
print("=" * 70)

f_direct = lambda x: mp.log(x, 2)
for n in [25]:
    mono = chebyshev_to_monomial(chebyshev_coefficients(f_direct, n, 2, 4), 2, 4)
    horner = list(reversed(mono))
    print(f"\nDegree {n-1}, Horner coefficients (f64):")
    for i, c in enumerate(horner):
        print(f"    {float(c):.20e},")

    # Check poly error (not fixed-point)
    max_err = mp.mpf(0)
    for i in range(100001):
        x = mp.mpf(2) + mp.mpf(2) * i / 100000
        y = mp.mpf(0)
        for c in horner:
            y = y * x + c
        err = abs(y - f_direct(x))
        if err > max_err: max_err = err
    bits = -float(mp.log(max_err, 2))
    print(f"\n  Polynomial approximation error: {bits:.2f} bits")
    print(f"  (Fixed-point will be ~16 bits due to coefficient cancellation)")

print()
print("=" * 70)
print("Option B: log2(1+t) on [0, 1], requires circuit change: t = a_norm/2 - 1")
print("=" * 70)

f_sub = lambda t: mp.log(1 + t, 2)
SCALE = mp.power(2, 63)

for n in [23]:
    mono = chebyshev_to_monomial(chebyshev_coefficients(f_sub, n, 0, 1), 0, 1)
    horner = list(reversed(mono))

    # Check polynomial error
    max_poly_err = mp.mpf(0)
    for i in range(100001):
        t = mp.mpf(i) / 100000
        y = mp.mpf(0)
        for c in horner:
            y = y * t + c
        err = abs(y - f_sub(t))
        if err > max_poly_err: max_poly_err = err
    poly_bits = -float(mp.log(max_poly_err, 2))

    # Check fixed-point error
    coef_q = [int(mp.nint(c * SCALE)) for c in horner]
    max_fp_err = mp.mpf(0)
    for i in range(10001):
        t = mp.mpf(i) / 10000
        t_q = int(mp.nint(t * SCALE))
        y = 0
        for c in coef_q:
            y = int(mp.mpf(y) * mp.mpf(t_q) / SCALE) + c
        result = mp.mpf(y) / SCALE
        err = abs(result - f_sub(t))
        if err > max_fp_err: max_fp_err = err
    fp_bits = -float(mp.log(max_fp_err, 2))

    print(f"\nDegree {n-1}, Horner coefficients (f64):")
    for i, c in enumerate(horner):
        print(f"    {float(c):.20e},")
    print(f"\n  Polynomial error: {poly_bits:.2f} bits")
    print(f"  Fixed-point error: {fp_bits:.2f} bits")

    # Also check: what if we use PRECISION_BITS=32?
    SCALE32 = mp.power(2, 32)
    coef_q32 = [int(mp.nint(c * SCALE32)) for c in horner]
    max_fp_err32 = mp.mpf(0)
    for i in range(10001):
        t = mp.mpf(i) / 10000
        t_q = int(mp.nint(t * SCALE32))
        y = 0
        for c in coef_q32:
            y = int(mp.mpf(y) * mp.mpf(t_q) / SCALE32) + c
        result = mp.mpf(y) / SCALE32
        err = abs(result - f_sub(t))
        if err > max_fp_err32: max_fp_err32 = err
    fp_bits32 = -float(mp.log(max_fp_err32, 2))
    print(f"  Fixed-point error (32-bit): {fp_bits32:.2f} bits")

# Also check the old polynomial on [2,4] with 63-bit fixed point
print()
print("=" * 70)
print("Old polynomial (degree 14) on [2, 4] — current code")
print("=" * 70)

old_coef = [
    -3.319586265362338e-08, 1.4957235315170112e-06,
    -3.1350053389526744e-05, 0.00040554177582512901,
    -0.0036218342998850703, 0.023663846121538389,
    -0.11691877183255484, 0.44524062371564499,
    -1.3195777548208449, 3.0518128028712077,
    -5.4904626000399528, 7.6298580090181591,
    -8.1653313719804235, 7.1389971101896279,
    -3.1937385492842112,
]

for pb in [32, 63]:
    S = mp.power(2, pb)
    cq = [int(mp.nint(mp.mpf(str(c)) * S)) for c in old_coef]
    max_e = mp.mpf(0)
    for i in range(10001):
        x = mp.mpf(2) + mp.mpf(2) * i / 10000
        x_q = int(mp.nint(x * S))
        y = 0
        for c in cq:
            y = int(mp.mpf(y) * mp.mpf(x_q) / S) + c
        result = mp.mpf(y) / S
        err = abs(result - f_direct(x))
        if err > max_e: max_e = err
    b = -float(mp.log(max_e, 2)) if max_e > 0 else 999
    print(f"  PRECISION_BITS={pb}: {b:.2f} bits")
