#!/usr/bin/env python3
"""Verify the degree-24 log2 polynomial with fixed-point arithmetic simulation."""

import mpmath as mp
mp.mp.dps = 80

PRECISION_BITS = 63
SCALE = mp.mpf(2) ** PRECISION_BITS

coef_f64 = [
    -4.15418640945004224513e-13, 3.12077228397159577428e-11, -1.12289661138612114666e-09,
    2.57576065557480061895e-08, -4.22955266934542677135e-07, 5.29229389897309921261e-06,
    -5.24453449601305460000e-05, 4.22323069047077573335e-04, -2.81336996113622185220e-03,
    1.57027573487949455300e-02, -7.40997267094792577691e-02, 2.97482257620215240213e-01,
    -1.02010087577349750632e+00, 2.99403111866345206238e+00, -7.52358910360607158196e+00,
    1.61630577745861039318e+01, -2.95902093561856709414e+01, 4.59245092142718647210e+01,
    -5.99779260786087320412e+01, 6.52609693530778969262e+01, -5.84107954746049529149e+01,
    4.23805613582126810002e+01, -2.47032331693540001538e+01, 1.22119558254723692414e+01,
    -3.94587506008177468786e+00,
]

def quantize(c):
    """Simulate quantization: round(c * 2^63)"""
    return int(round(mp.mpf(c) * SCALE))

coef_q = [quantize(c) for c in coef_f64]

def qmul(a, b):
    """Fixed-point multiply: (a * b) / 2^63, truncated."""
    return int(mp.mpf(a) * mp.mpf(b) / SCALE)

def qadd(a, b):
    return a + b

def horner_fixed(coef_q, x_q):
    """Horner evaluation in fixed-point."""
    y = 0
    for c in coef_q:
        y = qadd(qmul(y, x_q), c)
    return y

max_err = mp.mpf(0)
max_err_bits = 0
worst_x = 0

n_points = 10001
for i in range(n_points):
    x = mp.mpf(2) + mp.mpf(2) * i / (n_points - 1)
    x_q = int(round(x * SCALE))

    result_q = horner_fixed(coef_q, x_q)
    result = mp.mpf(result_q) / SCALE

    exact = mp.log(x, 2)
    err = abs(result - exact)

    if err > max_err:
        max_err = err
        worst_x = float(x)

bits = -float(mp.log(max_err, 2)) if max_err > 0 else 999
print(f"Fixed-point simulation (PRECISION_BITS={PRECISION_BITS}):")
print(f"  max abs error: {float(max_err):.6e}")
print(f"  precision bits: {bits:.2f}")
print(f"  worst at x = {worst_x:.6f}")
print(f"  target: >= 63 bits")
print(f"  {'PASS' if bits >= 63 else 'FAIL'}: {'sufficient' if bits >= 63 else 'insufficient'} for 63.63 fixed point")

# Also check old polynomial
old_coef_f64 = [
    -3.319586265362338e-08, 1.4957235315170112e-06,
    -3.1350053389526744e-05, 0.00040554177582512901,
    -0.0036218342998850703, 0.023663846121538389,
    -0.11691877183255484, 0.44524062371564499,
    -1.3195777548208449, 3.0518128028712077,
    -5.4904626000399528, 7.6298580090181591,
    -8.1653313719804235, 7.1389971101896279,
    -3.1937385492842112,
]
old_coef_q = [quantize(c) for c in old_coef_f64]

max_err_old = mp.mpf(0)
for i in range(n_points):
    x = mp.mpf(2) + mp.mpf(2) * i / (n_points - 1)
    x_q = int(round(x * SCALE))
    result_q = horner_fixed(old_coef_q, x_q)
    result = mp.mpf(result_q) / SCALE
    exact = mp.log(x, 2)
    err = abs(result - exact)
    if err > max_err_old:
        max_err_old = err

bits_old = -float(mp.log(max_err_old, 2)) if max_err_old > 0 else 999
print(f"\nOld polynomial (degree 14):")
print(f"  max abs error: {float(max_err_old):.6e}")
print(f"  precision bits: {bits_old:.2f}")
