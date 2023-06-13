# coding: utf-8
# 32.32 fixed point decimal with its arithmetic (mul and exp)
# [ref] https://github.com/XMunkki/FixPointCS/blob/c701f57c3cfe6478d1f6fd7578ae040c59386b3d/Cpp/Fixed64.h
# [ref] https://github.com/abdk-consulting/abdk-libraries-solidity/blob/master/ABDKMath64x64.sol
import math
RCP_LN2 = 0x171547652

def Qmul30(a, b):
    return a * b >> 30;

# precision: 18.19 bits
def Exp2Poly4(a):
    y = Qmul30(a, 14555373)
    y = Qmul30(a, y + 55869331)
    y = Qmul30(a, y + 259179547)
    y = Qmul30(a, y + 744137573)
    y = y + 1073741824
    return y

def Exp2Fast(x):
    k = int(x) % 0x100000000
    k = int(k / 0x4)
    y = Exp2Poly4(k)
    y = y * 4
    intPart = int(x / 0x100000000)
    # You must use lookup table to implement 2**intPart for intPart in Z[-32, 32]
    intPart = 2**intPart
    tmp = y * intPart
    return tmp

def Mul(a, b):
    return a * b / 0x100000000

def Exp(x):
    tmp = Mul(x, RCP_LN2)
    tmp = int(Exp2Fast(tmp))
    print('exp result {} {:02x}'.format(tmp, int(tmp)))
    return tmp

abs = lambda x: x if x >= 0 else -x
err = lambda x: abs((Exp(x * 0x100000000) / 0x100000000) - math.exp(x))

print(err(1))
print(err(-1.2))
print(err(-10))
print(err(-6))
print(err(10))
print(err(13))
print(err(14))
