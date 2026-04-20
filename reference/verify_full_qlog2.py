#!/usr/bin/env python3
"""Verify the full qlog2 function with the new polynomial and substitution."""
import mpmath as mp
import math

mp.mp.dps = 80

for PRECISION_BITS in [32, 63]:
    SCALE = mp.power(2, PRECISION_BITS)

    # New polynomial: log2(1+t) on [0,1], degree 22 (Horner, highest first)
    new_horner_f64 = [
        -1.61458691490921297314e-05, 2.02916186262885195075e-04,
        -1.21474871948364852685e-03, 4.62868649876543684918e-03,
        -1.26812473338416137336e-02, 2.68294268504126534602e-02,
        -4.61501662658178324339e-02, 6.74182317764813743288e-02,
        -8.71331474506971542793e-02, 1.03612055120922239015e-01,
        -1.17406657108512488263e-01, 1.30271944265860500911e-01,
        -1.44046233352145142126e-01, 1.60254402294701087106e-01,
        -1.80329761744040112381e-01, 2.06098430296388690497e-01,
        -2.40449096052631400289e-01, 2.88539003212634648232e-01,
        -3.60673760007960308993e-01, 4.80898346957287337045e-01,
        -7.21347520444400980288e-01, 1.44269504088896294292e+00,
        0.0
    ]

    # Old polynomial: log2(x) on [2,4], degree 14
    old_horner_f64 = [
        -3.319586265362338e-08, 1.4957235315170112e-06,
        -3.1350053389526744e-05, 0.00040554177582512901,
        -0.0036218342998850703, 0.023663846121538389,
        -0.11691877183255484, 0.44524062371564499,
        -1.3195777548208449, 3.0518128028712077,
        -5.4904626000399528, 7.6298580090181591,
        -8.1653313719804235, 7.1389971101896279,
        -3.1937385492842112,
    ]

    def quantize(c):
        return int(mp.nint(mp.mpf(c) * SCALE))

    def qmul(a, b):
        """Floor division like signed_div_scale for positive values."""
        return int(mp.floor(mp.mpf(a) * mp.mpf(b) / SCALE))

    def qadd(a, b):
        return a + b

    def horner_fp(coef_q, x_q):
        y = 0
        for c in coef_q:
            y = qmul(y, x_q) + c
        return y

    # Test x values: various positive numbers
    test_x = [0.25, 0.5, 1.0, 1.128, 2.0, 4.0, 10.0, 100.0, 0.01, 0.785398]

    new_coef_q = [quantize(c) for c in new_horner_f64]
    old_coef_q = [quantize(c) for c in old_horner_f64]

    print(f"\n{'='*70}")
    print(f"PRECISION_BITS = {PRECISION_BITS}")
    print(f"{'='*70}")

    max_err_new = mp.mpf(0)
    max_err_old = mp.mpf(0)

    for x_real in test_x:
        x_real = mp.mpf(x_real)
        exact = mp.log(x_real, 2)

        # Quantize x
        x_q = int(mp.nint(x_real * SCALE))

        # Find num_digits (highest set bit position)
        num_digits = 0
        temp = x_q
        for bit in range(256):
            if (temp >> bit) & 1:
                num_digits = bit

        # exp1 = num_digits, exp2 = num_digits + 1
        # shift = PRECISION_BITS + 2 - exp2 = PRECISION_BITS + 1 - num_digits
        shift_val = int(PRECISION_BITS) + 1 - num_digits

        # Normalize: a_norm should be in [2*SCALE, 4*SCALE)
        if shift_val >= 0:
            a_norm = x_q << shift_val
        else:
            a_norm = x_q >> (-shift_val)

        # === NEW approach: t = a_norm/2 - SCALE, then poly(t) + 1 ===
        a_norm_half = a_norm // 2
        t_q = a_norm_half - int(SCALE)
        log_1_plus_t = horner_fp(new_coef_q, t_q)
        log_a_norm_new = log_1_plus_t + int(SCALE)  # add 1 in fixed-point

        # log_shift = -shift_val, in fixed-point: -shift_val * SCALE
        # But shift in the code is (PRECISION_BITS + 2) - exp2, and log_shift = -shift
        # In the code: shift = PRECISION_BITS + 2 - (num_digits + 1) = PRECISION_BITS + 1 - num_digits
        # log_shift = -(shift), log_shift_q = -shift * SCALE
        log_shift_q = -shift_val * int(SCALE)
        result_new_q = log_a_norm_new + log_shift_q
        result_new = mp.mpf(result_new_q) / SCALE

        # === OLD approach: poly directly on a_norm in [2,4] ===
        log_a_norm_old = horner_fp(old_coef_q, a_norm)
        result_old_q = log_a_norm_old + log_shift_q
        result_old = mp.mpf(result_old_q) / SCALE

        err_new = abs(result_new - exact)
        err_old = abs(result_old - exact)
        if err_new > max_err_new: max_err_new = err_new
        if err_old > max_err_old: max_err_old = err_old

        print(f"  log2({float(x_real):10.6f}) exact={float(exact):12.8f}  "
              f"new={float(result_new):12.8f} err={float(err_new):.3e}  "
              f"old={float(result_old):12.8f} err={float(err_old):.3e}")

    bits_new = -float(mp.log(max_err_new, 2)) if max_err_new > 0 else 999
    bits_old = -float(mp.log(max_err_old, 2)) if max_err_old > 0 else 999
    print(f"\n  Max error NEW: {float(max_err_new):.6e} ({bits_new:.2f} bits)")
    print(f"  Max error OLD: {float(max_err_old):.6e} ({bits_old:.2f} bits)")
