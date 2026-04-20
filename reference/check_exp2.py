#!/usr/bin/env python3
"""Check if exp2 polynomial actually achieves 64 bits in fixed-point sim."""
import mpmath as mp
mp.mp.dps = 80

PRECISION_BITS = 63
SCALE = mp.power(2, PRECISION_BITS)

# exp2 coefficients from the Rust code (degree 12, on [0,1])
exp2_coef_f64 = [
    3.6240421303547230336183979205877e-11, 4.1284327467833130245549169910389e-10,
    0.0000000071086385644026346316624185550542, 0.00000010172297085296590958930245291448,
    0.0000013215904023658396206789543841996, 0.000015252713316417140696221389106544,
    0.00015403531076657894204857389177279, 0.0013333558131297097698435464957392,
    0.0096181291078409107025643582456283, 0.055504108664804181586140094858174,
    0.24022650695910142332414229540187, 0.69314718055994529934452147700678,
    1.0
]

# These are in order c0, c1, ..., c12 for Horner: y = c0*x^12 + c1*x^11 + ... + c12
# Wait, looking at the Rust code, the polynomial function does Horner as:
# y = ((c0 * x + c1) * x + c2) * x + ... + c_last
# So coef[0] is the highest degree coefficient
# Let me verify by checking: exp2(0) should be 1.0, and the last coef is 1.0

coef_q = [int(mp.nint(mp.mpf(str(c)) * SCALE)) for c in exp2_coef_f64]

max_err = mp.mpf(0)
for i in range(10001):
    t = mp.mpf(i) / 10000  # t in [0, 1]
    t_q = int(mp.nint(t * SCALE))
    y = 0
    for c in coef_q:
        y = int(mp.mpf(y) * mp.mpf(t_q) / SCALE) + c
    result = mp.mpf(y) / SCALE
    exact = mp.power(2, t)
    err = abs(result - exact)
    if err > max_err: max_err = err

bits = -float(mp.log(max_err, 2)) if max_err > 0 else 999
print(f"exp2 (degree 12) fixed-point simulation:")
print(f"  max error: {float(max_err):.6e}")
print(f"  precision bits: {bits:.2f}")
