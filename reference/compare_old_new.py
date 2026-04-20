#!/usr/bin/env python3
"""Compare old vs new on comprehensive test."""
import mpmath as mp
import random

mp.mp.dps = 80
random.seed(42)

for PRECISION_BITS in [32, 63]:
    SCALE = mp.power(2, PRECISION_BITS)

    # OLD polynomial on [2,4]
    old_horner = [
        -3.319586265362338e-08, 1.4957235315170112e-06,
        -3.1350053389526744e-05, 0.00040554177582512901,
        -0.0036218342998850703, 0.023663846121538389,
        -0.11691877183255484, 0.44524062371564499,
        -1.3195777548208449, 3.0518128028712077,
        -5.4904626000399528, 7.6298580090181591,
        -8.1653313719804235, 7.1389971101896279,
        -3.1937385492842112,
    ]
    old_cq = [int(mp.nint(mp.mpf(c) * SCALE)) for c in old_horner]

    # NEW polynomial log2(1+t) on [0,1]
    new_horner = [
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
    new_cq = [int(mp.nint(mp.mpf(c) * SCALE)) for c in new_horner]

    def qmul(a, b):
        return int(mp.floor(mp.mpf(a) * mp.mpf(b) / SCALE))

    def horner_fp(cq, x_q):
        y = 0
        for c in cq:
            y = qmul(y, x_q) + c
        return y

    test_x = []
    for e in range(-10, 11):
        test_x.append(mp.power(2, e))
    for _ in range(1000):
        test_x.append(mp.mpf(random.uniform(0.001, 1000.0)))

    max_err_old = mp.mpf(0)
    max_err_new = mp.mpf(0)

    for x_real in test_x:
        if x_real <= 0: continue
        exact = mp.log(x_real, 2)
        x_q = int(mp.nint(x_real * SCALE))
        if x_q <= 0: continue

        num_digits = 0
        for bit in range(256):
            if (x_q >> bit) & 1:
                num_digits = bit

        shift_val = int(PRECISION_BITS) + 1 - num_digits
        if shift_val >= 0:
            a_norm = x_q << shift_val
        else:
            a_norm = x_q >> (-shift_val)

        # OLD: poly directly on a_norm
        log_old = horner_fp(old_cq, a_norm)
        log_shift_q = -shift_val * int(SCALE)
        result_old = mp.mpf(log_old + log_shift_q) / SCALE
        err_old = abs(result_old - exact)
        if err_old > max_err_old: max_err_old = err_old

        # NEW: substitution
        t_q = a_norm // 2 - int(SCALE)
        log_new = horner_fp(new_cq, t_q) + int(SCALE)
        result_new = mp.mpf(log_new + log_shift_q) / SCALE
        err_new = abs(result_new - exact)
        if err_new > max_err_new: max_err_new = err_new

    bits_old = -float(mp.log(max_err_old, 2)) if max_err_old > 0 else 999
    bits_new = -float(mp.log(max_err_new, 2)) if max_err_new > 0 else 999
    print(f"P={PRECISION_BITS}: OLD={bits_old:.2f} bits ({float(max_err_old):.3e})  NEW={bits_new:.2f} bits ({float(max_err_new):.3e})")
