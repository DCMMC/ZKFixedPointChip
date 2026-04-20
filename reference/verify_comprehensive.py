#!/usr/bin/env python3
"""Comprehensive test of HP qlog2 and qexp2 — measures circuit accuracy vs ideal quantized output."""
import mpmath as mp
import random

mp.mp.dps = 100
random.seed(42)

K = 20  # EXTRA_PRECISION_BITS

LOG2_COEF_HP = {
    32: [
        -72714530283, 913853260841, -5470741880415, 20845750791055,
        -57111260767282, 120828996766082, -207841871597824,
        303624723506539, -392412810390578, 466627212833677,
        -528752577204712, 586692679652559, -648726562848844,
        721721666458897, -812133047794269, 928184813884460,
        -1082886459384203, 1299464147350276, -1624330211174166,
        2165773616159927, -3248660424278035, 6497320848556796, 0,
    ],
    63: [
        8489630466307435614, -136264558136253235350,
        1050526704828427793816, -5189834027395357318753,
        18506692453020330767193, -50926565676127844703261,
        113060535939451201505032, -209440667972691351779979,
        333122647536372942841436, -467190200783459387677135,
        593095012913227390548973, -699534786007381648647137,
        785655430331027083568081, -858016718324957451357692,
        925384172747064501184695, -995233525936403540442198,
        1072954147630102434451654, -1162669663058552030975272,
        1268432389421105617469566, -1395287320139384762195722,
        1550320948804107476170624, -1744111267449001152614350,
        1993270038527229370667010, -2325481712951963447868217,
        2790578055614722117309453, -3488222569521243587856957,
        4650963426028401854480036, -6976445139042604142949689,
        13952890278085208301077931, 14,
    ],
}

EXP2_COEF_HP = {
    32: [
        163898, 1855089, 32025673, 458102292, 5951930881,
        68692103124, 693713372880, 6004900741814, 43316202666321,
        249968283100316, 1081884007225527, 3121657384082680,
        4503599627370496,
    ],
    63: [
        1859160554, 28050891277, 660997987636, 13232945686538,
        248356396920884, 4299452683312243, 68230923496384621,
        984364075377840504, 12781234560285917319,
        147515389723298731308, 1489738048462252400249,
        12895426168440252968521, 93020836916792629615618,
        536802800476790956263267, 2323328214549521983009611,
        6703708186976009930559257, 9671406556917033397649408,
    ],
}

def horner_hp_exp2(coef_q, x_q, PRECISION_BITS):
    """HP Horner for exp2: x at P-bit scale, internally scales to HP."""
    HP_SCALE = mp.power(2, PRECISION_BITS + K)
    K_SCALE = int(mp.power(2, K))
    x_hp = x_q * K_SCALE
    y = 0
    for ci, c in enumerate(coef_q):
        y_add = y + c
        if ci < len(coef_q) - 1:
            y = int(mp.floor(mp.mpf(y_add) * mp.mpf(x_hp) / HP_SCALE))
        else:
            y = int(mp.floor(mp.mpf(y_add) / K_SCALE))
    return y

def simulate_qlog2(x_q, PRECISION_BITS, coef_q):
    """Simulate the full HP qlog2 pipeline matching the Rust code."""
    SCALE = int(mp.power(2, PRECISION_BITS))
    HP_SCALE = int(mp.power(2, PRECISION_BITS + K))
    K_SCALE = int(mp.power(2, K))
    HP_QUANT_SCALE = HP_SCALE  # 2^{P+K}

    # Find highest bit
    num_digits = 0
    for bit in range(256):
        if (x_q >> bit) & 1:
            num_digits = bit

    # Scale up to HP
    a_hp = x_q * K_SCALE

    # Shift: same as original (P+2) - exp2
    exp2_val = num_digits + 1
    shift_val = PRECISION_BITS + 2 - exp2_val

    if shift_val >= 0:
        a_hp_norm = a_hp << shift_val
    else:
        a_hp_norm = a_hp >> (-shift_val)

    # t_hp = a_hp_norm/2 - 2^{P+K}
    a_hp_norm_half = a_hp_norm // 2
    t_hp = a_hp_norm_half - HP_QUANT_SCALE

    # Horner at HP scale
    y = 0
    for ci, c in enumerate(coef_q):
        y_add = y + c
        if ci < len(coef_q) - 1:
            y = int(mp.floor(mp.mpf(y_add) * mp.mpf(t_hp) / HP_SCALE))
        else:
            y = y_add
    log_1_plus_t_hp = y

    # log2(a_hp_norm_real) = 1 + poly(t)
    log_a_norm_hp = log_1_plus_t_hp + HP_QUANT_SCALE

    # Shift correction
    log_shift_q = -shift_val * HP_QUANT_SCALE
    result_hp = log_a_norm_hp + log_shift_q

    # Convert to P-bit scale
    result = int(mp.floor(mp.mpf(result_hp) / K_SCALE))
    return result

for PRECISION_BITS in [32, 63]:
    SCALE = int(mp.power(2, PRECISION_BITS))

    test_x = []
    for e in range(-10, 11):
        test_x.append(mp.power(2, e))
    for _ in range(1000):
        test_x.append(mp.mpf(random.uniform(0.001, 1000.0)))
    for e in range(-5, 10):
        base = mp.power(2, e)
        for delta in [0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999]:
            test_x.append(base * (1 + delta))

    # Test log2
    coef_q = LOG2_COEF_HP[PRECISION_BITS]
    max_err_log = mp.mpf(0)
    max_ulp_log = 0

    for x_real in test_x:
        x_real = mp.mpf(x_real)
        if x_real <= 0: continue
        x_q = int(mp.nint(x_real * SCALE))
        if x_q <= 0: continue
        x_quantized = mp.mpf(x_q) / SCALE
        exact = mp.log(x_quantized, 2)

        result_q = simulate_qlog2(x_q, PRECISION_BITS, coef_q)
        result = mp.mpf(result_q) / SCALE

        err = abs(result - exact)
        ideal_q = int(mp.nint(exact * SCALE))
        ulp = abs(result_q - ideal_q)
        if err > max_err_log: max_err_log = err
        if ulp > max_ulp_log: max_ulp_log = int(ulp)

    bits_log = -float(mp.log(max_err_log, 2)) if max_err_log > 0 else 999

    # Test exp2
    exp2_coef = EXP2_COEF_HP[PRECISION_BITS]
    max_err_exp = mp.mpf(0)
    max_ulp_exp = 0
    for i in range(50001):
        t_q = int(mp.mpf(i) * SCALE / 50000)
        y = horner_hp_exp2(exp2_coef, t_q, PRECISION_BITS)
        t_quantized = mp.mpf(t_q) / SCALE
        exact = mp.power(2, t_quantized)
        result = mp.mpf(y) / SCALE
        err = abs(result - exact)
        ideal_q = int(mp.nint(exact * SCALE))
        ulp = abs(y - ideal_q)
        if err > max_err_exp: max_err_exp = err
        if ulp > max_ulp_exp: max_ulp_exp = int(ulp)

    bits_exp = -float(mp.log(max_err_exp, 2)) if max_err_exp > 0 else 999

    status_log = "PASS" if bits_log >= PRECISION_BITS else "FAIL"
    status_exp = "PASS" if bits_exp >= PRECISION_BITS else "FAIL"
    print(f"PRECISION_BITS={PRECISION_BITS}:")
    print(f"  log2: {bits_log:.2f} bits, max {max_ulp_log} ULP [{status_log}]")
    print(f"  exp2: {bits_exp:.2f} bits, max {max_ulp_exp} ULP [{status_exp}]")
