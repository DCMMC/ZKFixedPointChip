#!/usr/bin/env python3
"""Generate polynomial for log2(1+t) on [0, 1] via Chebyshev interpolation.
This is used for log2(x) on [2, 4] via the substitution t = x/2 - 1."""

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

# Approximate log2(1+t) on [0, 1]
f = lambda t: mp.log(1 + t, 2)
a, b = mp.mpf(0), mp.mpf(1)

print("=== Polynomial for log2(1+t) on [0, 1] ===")
print("(For log2(x) on [2,4]: set t = x/2 - 1, then log2(x) = 1 + log2(1+t))")
print()

for n in range(10, 30):
    cheb_coeffs = chebyshev_coefficients(f, n, a, b)
    mono_coeffs = chebyshev_to_monomial(cheb_coeffs, a, b)
    max_err = mp.mpf(0)
    for i in range(10001):
        t = mp.mpf(i) / 10000
        err = abs(eval_poly(mono_coeffs, t) - f(t))
        if err > max_err:
            max_err = err
    bits = -float(mp.log(max_err, 2))
    print(f"  n={n} (degree {n-1}): {bits:.2f} bits (max error: {float(max_err):.6e})")

    if bits >= 66:
        degree = n - 1
        print(f"\n==> degree {degree} achieves {bits:.2f} bits")

        # Check coefficient magnitudes
        print(f"\nCoefficient magnitudes:")
        for i in range(n):
            print(f"  c[{i}] = {float(mono_coeffs[i]):.20e}  (|c| = {float(abs(mono_coeffs[i])):.6e})")

        # Horner coefficients (highest degree first)
        print(f"\nHorner coefficients (x^{degree} to x^0):")
        horner = []
        for i in reversed(range(n)):
            horner.append(mono_coeffs[i])

        # Now simulate fixed-point evaluation
        PRECISION_BITS = 63
        SCALE = mp.mpf(2) ** PRECISION_BITS

        def quantize(c):
            return int(mp.nint(mp.mpf(c) * SCALE))

        coef_q = [quantize(float(c)) for c in horner]

        def qmul_sim(a, b):
            return int(mp.mpf(a) * mp.mpf(b) / SCALE)

        max_err_fp = mp.mpf(0)
        for i in range(10001):
            t = mp.mpf(i) / 10000
            t_q = int(mp.nint(t * SCALE))
            # Horner in fixed-point
            y = 0
            for c in coef_q:
                y = qmul_sim(y, t_q) + c
            result = mp.mpf(y) / SCALE
            exact = f(t)
            err = abs(result - exact)
            if err > max_err_fp:
                max_err_fp = err

        bits_fp = -float(mp.log(max_err_fp, 2)) if max_err_fp > 0 else 999
        print(f"\nFixed-point simulation (63-bit):")
        print(f"  max abs error: {float(max_err_fp):.6e}")
        print(f"  precision bits: {bits_fp:.2f}")
        print(f"  {'PASS' if bits_fp >= 63 else 'FAIL'}")

        if bits_fp >= 63:
            print(f"\nRust code for generate_log_poly:")
            print(f"    fn generate_log_poly(&self) -> Vec<QuantumCell<F>> {{")
            print(f"        // Chebyshev approx of log2(1+t) on [0,1], degree {degree}, precision: {bits:.2f} bits")
            print(f"        // Used as: t = a_norm / 2 - 1, log2(a_norm) = 1 + poly(t)")
            print(f"        // Estimated max error: {float(max_err):.20e}")
            print(f"        let coef: Vec<F> = [")
            line = "            "
            for i, c in enumerate(horner):
                s = f"{float(c):.20e}"
                if i < len(horner) - 1:
                    s += ", "
                if len(line) + len(s) > 100:
                    print(line.rstrip())
                    line = "            " + s
                else:
                    line += s
            print(line.rstrip())
            print(f"        ].into_iter().map(|c| self.quantization(c)).collect();")
            print(f"        coef.iter().map(|x| Constant(*x)).collect()")
            print(f"    }}")

        break
