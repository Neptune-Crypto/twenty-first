def cyclic_mul(a, b):
    prod = [0 for i in range(len(a)+len(b)-1)]
    for i in range(len(a)):
        for j in range(len(b)):
            prod[i+j] += a[i] * b[j]
    assert(len(a) == len(b))
    for i in range(len(a)-1):
        prod[i] += prod[len(a)+i]
    res = prod[:len(a)]
    return res

def cyclic_mul8_i64(a, b):
    n = 8
    assert(n==len(a))
    assert(n==len(b))
    for aa in a:
        assert(aa >= -2^63), "a is too small"
        assert(aa < 2^63), "a is too large"
    for bb in b:
        assert(bb >= -2^63), "b is too small"
        assert(bb < 2^63), "b is too large"
    prod = [0 for i in range(len(a)+len(b)-1)]
    for i in range(len(a)):
        for j in range(len(b)):
            prod[i+j] += a[i] * b[j]
    assert(len(a) == len(b))
    for i in range(len(a)-1):
        prod[i] += prod[len(a)+i]
    res = prod[:len(a)]
    for r in res:
        assert(r >= -2^63), "r is too small"
        assert(r < 2^63), "r is too large"
    return res

def negacyclic_mul(a,b):
    prod = [0 for i in range(len(a)+len(b)-1)]
    for i in range(len(a)):
        for j in range(len(b)):
            prod[i+j] += a[i] * b[j]
    assert(len(a) == len(b))
    for i in range(len(a)-1):
        prod[i] -= prod[len(a)+i]
    res = prod[:len(a)]
    return res

import random

def test_decomposition():
    n = 4;
    a = [random.choice(range(10)) for i in range(n)]
    b = [random.choice(range(10)) for i in range(n)];

    ahi_plus_alo = [a[i] + a[i+n//2] for i in range(n//2)]
    ahi_minus_alo = [a[i] - a[i+n//2] for i in range(n//2)]

    bhi_plus_blo = [b[i] + b[i+n//2] for i in range(n//2)]
    bhi_minus_blo = [b[i] - b[i+n//2] for i in range(n//2)]

    cyc = cyclic_mul(ahi_plus_alo, bhi_plus_blo)
    neg = negacyclic_mul(ahi_minus_alo, bhi_minus_blo)
    hi = [cyc[i] + neg[i] for i in range(n//2)]
    lo = [cyc[i] - neg[i] for i in range(n//2)]
    res = hi + lo
    for i in range(len(res)):
        res[i] //= 2

    assert(res == cyclic_mul(a,b))

p = (1<<64) - (1<<32) + 1
field = FiniteField(p)

def negacyclic_mul8(a,b):
    n = 8
    xi = field(1<<12)
    a = [a[i] * xi^i for i in range(n)]
    b = [b[i] * xi^i for i in range(n)]

    c = cyclic_mul(a,b)

    c = [c[i] / xi^i for i in range(n)]
    return c

def test_negacyclic_mul8():
    n = 8;
    a = [field.random_element() for i in range(n)]
    b = [field.random_element() for i in range(n)]

    c_ = negacyclic_mul(a,b)
    c = negacyclic_mul8(a,b)

    assert(c_ == c)

def sl32(aa):
    ofl = aa >> 32
    return ((aa << 32) & 0xffffffffffffffff) + (ofl << 32) - ofl

def shift_left_reduce(aa, shamt):
    assert(aa>=0), "shift_left_reduce only works for positive integers"
    assert(aa<2^64), "shift_left_reduce only works for integers < 2^64"
    assert(0<=shamt and shamt < 128), "shift_left_reduce only works for 0 <= shamt < 128"
    if shamt >= 64:
        a = sl32(aa) - aa
        sh = shamt - 64
    else:
        a = aa
        sh = shamt
    ofl = a >> (64 - sh)
    b = ((a << sh) & 0xffffffffffffffff) + sl32(ofl) - ofl
    if b > 2^64:
        return b - p
    else:
        return b

def smart_cyclomul16(a, b):
    n = 16
    assert(n==len(a))
    assert(n==len(b))
 
    ahi = [a[i] for i in range(n//2)]
    alo = [a[i+n//2] for i in range(n//2)]
    ahi_plus_alo = [a[i] + a[i+n//2] for i in range(n//2)]
    ahi_minus_alo = [a[i] - a[i+n//2] for i in range(n//2)]
    ahi_minus_alo = [aa + p if aa < 0 else aa for aa in ahi_minus_alo]
    assert(all(am >= 0 and am < 2^64 for am in ahi_minus_alo)), "am not as expected"

    # 32768, 8, 16384, 1024, 512, 128, 2048, 8
    # 16384, 256, 512, 1024, 8, 1024, 1024, 4096
    bhi_plus_blo = [b[i] + b[i+n//2] for i in range(n//2)]
    bhi_minus_blo = [b[i] - b[i+n//2] for i in range(n//2)]
    bhi_plus_blo = [32768, 8, 16384, 1024, 512, 128, 2048, 8]

    pluses = cyclic_mul8_i64(ahi_plus_alo, bhi_plus_blo)

    am = [shift_left_reduce(ahi_minus_alo[i], (12 * i)) for i in range(n//2)]
    assert(all(am_ >= 0 for am_ in am)), "am_ not positive"
    assert(all(am_ < 2^64 for am_ in am)), "am_ too large"
    bm = [bhi_minus_blo[i] << (12 * i) for i in range(n//2)]
    bm = [16384, 256, 512, 1024, 8, 1024, 1024, 4096]

    minuses = cyclic_mul(am, bm)

    minuses = [shift_left_reduce(minuses[i], 12 * (n//2-i)) for i in range(n//2)]
    
    sums = [(-pluses[i] + minuses[i])/2 for i in range(n//2)]
    diffs = [(-pluses[i] - minuses[i])/2 for i in range(n//2)]

    return sums + diffs

def field_cyclomul16(a, b):
    n = 16
    assert(n==len(a))
    assert(n==len(b))

    ahi = [a[i] >> 32 for i in range(n)]
    alo = [a[i] & 0xffffffff for i in range(n)]

    chi = smart_cyclomul16(ahi, b)
    clo = smart_cyclomul16(alo, b)

    print("max: 2^", log(1.0*max(max(chi), max(clo)), 2))
    print("min: -2^", log(-1.0*min(min(chi), min(clo)), 2))

    c = [field((chi[i] << 32) + clo[i]) for i in range(n)]

    return c

def test_field_cyclomul16():
    n = 16
    a = [field.random_element() for i in range(n)]
    #b = [field(random.choice(range(1<<16))) for i in range(n)]
    #b = [28564, 47088, 31462, 12728, 17558, 1535, 14124, 1428, 21378, 20830, 57295, 20956, 32987, 11852, 30063, 15422]
    b = [24576, 17870283317245378565, 18446462594437947393, 18446743931975631393, 18445618169507741953, 18446708885042495553, 18446744060824650753, 18446744069406195717, 8192, 576460752169205764, 281474976653312, 137438953952, 1125899906842880, 35184372088896, 8589935616, 8388612]

    #print("b:", b)

    c_smart = [-e for e in field_cyclomul16([ZZ(a[i]) for i in range(n)],b)]
    c_regular = cyclic_mul(a, b)

    assert(c_smart == c_regular)

def sample_mds_constants():
    constants = [2^random.choice(range(16)) for i in range(16)]
    # 32768, 8, 16384, 1024, 512, 128, 2048, 8
    # 16384, 256, 512, 1024, 8, 1024, 1024, 4096
    constants = [32768, 8, 16384, 1024, 512, 128, 2048, 8, 16384, 256, 512, 1024, 8, 1024, 1024, 4096]
    constants = [field(c) for c in constants]
    print(constants[:8])
    print(constants[8:])

    shifted = [constants[8+i] / field(1 << (12 * i)) for i in range(8)]

    sums = [(constants[i] + shifted[i])/2 for i in range(8)]
    diffs = [(constants[i] - shifted[i])/2 for i in range(8)]    

    print("sums:", sums)
    print("diffs:", diffs)
    print("col:", sums+diffs)
    print("bins:", [bin(sd) for sd in sums+diffs])

