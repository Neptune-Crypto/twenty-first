def fft( vec, omega=None ):
    if len(vec) == 1:
        return vec

    field = vec[0].parent()
    if omega == None:
        omega = field(1<<12)

    n = omega.multiplicative_order()
    half = n // 2
    log2n = len(bin(n-1)[2:])
    while len(vec) < n:
        omega = omega^2
        n = n // 2
        half = half // 2
        log2n = log2n - 1

    assert(half > 0)

    evens = fft(vec[::2], omega^2)
    odds = fft(vec[1::2], omega^2)
    
    vv = [field(0) for i in range(n)]
    for i in range(n):
        vv[i] = evens[i%half] + omega^i * odds[i%half]

    return vv

def ifft( vec ):
    field = vec[0].parent()
    omega = field(1<<12)^-1
    return [v / 16 for v in fft(vec, omega)]

def bitreverse(l, k):
    acc = 0
    for i in range(k):
        if l & (1<<i) != 0:
            acc |= 1 << (k-1-i)
    return acc

def fft_noswap(vec):
    vv = fft(vec)
    for i in range(16):
        if i < bitreverse(i, 4):
            temp = vv[i]
            vv[i] = vv[bitreverse(i, 4)]
            vv[bitreverse(i, 4)] = temp
    return vec

def fft_noswap(vec):
    vv = [v for v in vec]
    for i in range(16):
        if i < bitreverse(i, 4):
            temp = vv[i]
            vv[i] = vv[bitreverse(i, 4)]
            vv[bitreverse(i, 4)] = temp
    return ifft(vv)

def fft2( vec ):
    for i in range(0, len(vec), 2):
        a = vec[i] + vec[i+1]
        b = vec[i] - vec[i+1]
        vec[i] = a
        vec[i+1] = b

def fft4( vec ):
    field = vec[0].parent()
    omega_16 = field(1<<12)
    omega_8 = omega_16^2
    omega_4 = omega_8^2

    for i in range(0, 16, 4):
        vv = [field(0) for j in range(4)]
        for j in range(2):
            vv[j] = vec[i+j] + vec[i+2+j]
            vv[2+j] = omega_4^j * vec[i+j] - omega_4^j * vec[i+2+j]
        vec[i:(i+4)] = vv[:]

def fft8( vec ):
    field = vec[0].parent()
    omega_16 = field(1<<12)
    omega_8 = omega_16^2

    vv = [field(0) for i in range(8)]

    for i in range(0, 16, 8):
        vv = [field(0) for j in range(8)]
        for j in range(4):
            vv[j] = vec[i+j] + vec[i+4+j]
            vv[j+4] = omega_8^j * vec[i+j] + omega_8^(4+j) * vec[i+4+j]
        vec[i:(i+8)] = vv[:]

def fft16( vec ):
    field = vec[0].parent()
    omega_16 = field(1<<12)

    vv = [field(0) for i in range(16)]

    for j in range(8):
        vv[j] = vec[j] + vec[8+j]
        vv[j+8] = omega_16^j * vec[j] + omega_16^(8+j) * vec[8+j]
    for j in range(16):
        vec[j] = vv[j] 

def ifft2(vec):
    fft2(vec)
    for i in range(len(vec)):
        vec[i] /= 2

def ifft4(vec):
    field = vec[0].parent()
    omega_16 = field(1<<12)^-1
    omega_8 = omega_16^2
    omega_4 = omega_8^2

    for i in range(0, len(vec), 4):
        vv = [field(0) for j in range(4)]
        for j in range(4):
            vv[j] = vec[i+(j%2)] + omega_4^j * vec[i+2+(j%2)]
        for j in range(4):
            vec[i+j] = vv[j]

    for i in range(len(vec)):
        vec[i] /= 2

def ifft8( vec ):
    field = vec[0].parent()
    omega_16 = field(1<<12)^-1
    omega_8 = omega_16^2

    for i in range(0, len(vec), 8):
        vv = [field(0) for j in range(8)]
        for j in range(8):
            vv[j] = vec[i+(j%4)] + omega_8^j * vec[i+4+(j%4)]
        for j in range(8):
            vec[i+j] = vv[j] / 2

def ifft16( vec ):
    field = vec[0].parent()
    omega_16 = field(1<<12)^-1

    vv = [field(0) for j in range(16)]
    for j in range(16):
        vv[j] = vec[(j%8)] + omega_16^j * vec[8+(j%8)]
    for j in range(16):
        vec[j] = vv[j] / 2

def test_inverses():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)
    
    vec = [field.random_element() for i in range(16)]
    vv = vec[:]

    fft2(vec)
    ifft2(vec)
    assert(vec == vv), f"fft2:\nvec: {vec}\nvv: {vv}"

    fft4(vec)
    ifft4(vec)
    assert(vec == vv), f"fft4:\nvec: {vec}\nvv: {vv}"
    fft8(vec)
    ifft8(vec)
    assert(vec == vv), f"fft8:\nvec: {vec}\nvv: {vv}"
    fft16(vec)
    ifft16(vec)
    assert(vec == vv), f"fft16:\nvec: {vec}\nvv: {vv}"

    assert(vec == ifft(fft(vec))), f"fft:\nvec: {vec}\nifft(fft(vec)): {ifft(fft(vec))}"

def cyclomul(f, g):
    field = f[0].parent()
    h = [field(0) for i in range(len(f))]
    for i, fi in enumerate(f):
        for j, gj in enumerate(g):
            h[(i+j)%len(f)] += fi*gj
    return h

def hadmul(f,g):
    return [fi * gi for fi, gi in zip(f, g)]

def test_cyclomul_fft():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    f = [field.random_element() for i in range(16)]
    g = [field.random_element() for i in range(16)]

    h = cyclomul(f,g)
    h_ = ifft(hadmul(fft(f), fft(g)))

    assert(h == h_), f"hadamard multiplication in freq domain does not equal cyclic multiplication in time domain"

def bitswap(vec):
    for i in range(16):
        if i < bitreverse(i, 4):
            temp = vec[i]
            vec[i] = vec[bitreverse(i, 4)]
            vec[bitreverse(i, 4)] = temp

def test_fft_decomposition():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)
    
    vec = [field.random_element() for i in range(16)]

    vv = fft(vec)

    fft16(vec)
    fft8(vec)
    fft4(vec)
    fft2(vec)
    bitswap(vec)

    assert(vec == vv), f"decomposed fft does not equal fft"

def test_cyclomul_decomposed_fft():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    f = [field.random_element() for i in range(16)]
    g = [field.random_element() for i in range(16)]

    h = cyclomul(f,g)

    fft16(f)
    fft8(f)
    fft4(f)
    fft2(f)
    bitswap(f)
    
    fft16(g)
    fft8(g)
    fft4(g)
    fft2(g)
    bitswap(g)
    
    h_ = hadmul(f,g)

    bitswap(h_)
    ifft2(h_)
    ifft4(h_)
    ifft8(h_)
    ifft16(h_)

    assert(h == h_), f"hadamard multiplication in freq domain does not equal cyclic multiplication in time domain"

def test_cyclomul_hybrid():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    f = [field.random_element() for i in range(16)]
    g = [field.random_element() for i in range(16)]

    h = cyclomul(f,g)

    fft16(f)
    #fft8(f)
    #fft4(f)
    #fft2(f)
    #bitswap(f)
    
    fft16(g)
    #fft8(g)
    #fft4(g)
    #fft2(g)
    #bitswap(g)
    
    #h_ = hadmul(f,g)
    h_ = [field(0) for i in range(16)]
    l = 8
    for i in range(0,16,l):
        hh = cyclomul(f[i:(i+l)], g[i:(i+l)])
        h_[i:(i+l)] = hh

    #bitswap(h_)
    #ifft2(h_)
    #ifft4(h_)
    #ifft8(h_)
    ifft16(h_)

    assert(h == h_), f"hadamard multiplication in freq domain does not equal cyclic multiplication in time domain"

def negacyclomul(f, g):
    field = f[0].parent()
    h = [field(0) for i in range(len(f))]
    for i, fi in enumerate(f):
        for j, gj in enumerate(g):
            if i+j < len(h):
                h[i+j] += fi*gj
            else:
                h[i+j-len(h)] -= fi*gj
    return h

def test_negacyclomul():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    omega = field(1<<12)

    f = [field.random_element() for i in range(8)]
    g = [field.random_element() for i in range(8)]

    ff = [omega^i * fi for i, fi in enumerate(f)]
    gg = [omega^i * gi for i, gi in enumerate(g)]

    hh = cyclomul(ff,gg)

    h = [omega^-i * hi for i, hi in enumerate(hh)]

    h_ = negacyclomul(f, g)

    assert(all(hi == hj for hi, hj in zip(h, h_))), f"negacyclomul not correct"

def karatsuba(f, g):
    n = len(f)
    half = n//2

    if n == 1:
        return [f[0] * g[0]]

    flo = f[:half]
    fhi = f[half:]
    glo = g[:half]
    ghi = g[half:]

    his = karatsuba(fhi, ghi)
    los = karatsuba(flo, glo)
    both = karatsuba([a + b for a, b in zip(fhi, flo)], [a + b for a, b in zip(ghi, glo)])

    field = f[0].parent()
    result = [field(0) for i in range(2*n-1)]
    for i in range(len(los)):
        result[i] += los[i]
    for i in range(len(his)):
        result[n+i] += his[i]
    for i in range(len(both)):
        result[half+i] += both[i] - his[i] - los[i]

    return result

def test_karatsuba():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    omega = field(1<<12)

    n = 8

    f = [field.random_element() for i in range(n)]
    g = [field.random_element() for i in range(n)]

    h = karatsuba(f,g)

    h_ = [field(0) for i in range(2*n-1)]
    for i in range(n):
        for j in range(n):
            h_[i+j] += f[i] * g[j]

    assert(h_ == h), f"karatsuba is wrong.\nh: {h}\nh_ {h_}"

def fast_negacyclomul(f,g):
    h = karatsuba(f,g)
    for i in range(len(f), len(h)):
        h[i-len(f)] -= h[i]

    return h[:len(f)]

def fast_negacyclomul8(f,g):
    n = 8
    half = 4

    flo = f[:half]
    fhi = f[half:]
    glo = g[:half]
    ghi = g[half:]

    his = karatsuba(fhi, ghi)
    los = karatsuba(flo, glo)
    both = karatsuba([a + b for a, b in zip(fhi, flo)], [a + b for a, b in zip(ghi, glo)])

    field = f[0].parent()
    result = [field(0) for i in range(2*n-1)]
    for i in range(len(los)):
        result[i] += los[i]
    for i in range(len(his)):
        result[n+i] += his[i]
    for i in range(len(both)):
        result[half+i] += both[i] - his[i] - los[i]

    for i in range(len(f), len(result)):
        result[i-len(f)] -= result[i]

    return result[:len(f)]

def fast_cyclomul16(f, g):
    field = f[0].parent()
    omega = field(1<<12)

    ff = [field(0) for i in range(16)]
    for i in range(8):
        ff[i] = f[i] + f[i+8]
    for i in range(8):
        ff[i+8] = f[i] - f[i+8]

    gg = [field(0) for i in range(16)]
    for i in range(8):
        gg[i] = g[i] + g[i+8]
    for i in range(8):
        gg[i+8] = g[i] - g[i+8]

    hh = cyclomul(ff[:8], gg[:8]) + fast_negacyclomul8(ff[8:], gg[8:])

    return [(hh[i] + hh[i+8])/2 for i in range(8)] + [(hh[i] - hh[i+8])/2 for i in range(8)]

def test_fast_cyclomul():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    f = [field.random_element() for i in range(16)]
    g = [field.random_element() for i in range(16)]

    h = cyclomul(f,g)

    h_ = fast_cyclomul16(f,g)

    assert(h_ == h), f"fastmul does not give same result as ordinary mul.\nh_: {h_}\nh:: {h}"

def complex_sum(f, g):
    return [(a[0] + b[0], a[1] + b[1]) for a, b in zip(f, g)]

def complex_diff(f, g):
    return [(a[0] - b[0], a[1] - b[1]) for a, b in zip(f, g)]

def complex_mul(a, b):
    lo = a[0] * b[0]
    hi = a[1] * b[1]
    li = (a[0] + a[1]) * (b[0] + b[1]) - lo - hi
    return (lo - hi, li)

def test_complex_mul():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    a = (field.random_element(), field.random_element())
    b = (field.random_element(), field.random_element())

    c = (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
    assert(c == complex_mul(a,b))

def complex_karatsuba(f, g):
    if len(f) == 1:
        return [complex_mul(f[0], g[0])]

    n = len(f)
    half = n//2

    lo = complex_karatsuba(f[:half], g[:half])
    hi = complex_karatsuba(f[half:], g[half:])
    ff = complex_sum(f[:half], f[half:])
    gg = complex_sum(g[:half], g[half:])

    li = complex_diff(complex_karatsuba(ff, gg), complex_sum(lo, hi))


    if len(f) == 4:
        print("lo:", lo)
        print("hi:", hi)
        print("li:", li)

    field = f[0][0].parent()
    result = [[field(0), field(0)] for i in range(2*n-1)]
    for i in range(len(lo)):
        result[i][0] += lo[i][0]
        result[i][1] += lo[i][1]
    for i in range(len(li)):
        result[half+i][0] += li[i][0]
        result[half+i][1] += li[i][1]
    for i in range(len(hi)):
        result[n+i][0] += hi[i][0]
        result[n+i][1] += hi[i][1]

    return [tuple(r) for r in result]

def test_complex_karatsuba():
    
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    f = [(field.random_element(), field.random_element()) for i in range(4)]
    g = [(field.random_element(), field.random_element()) for i in range(4)]

    h = complex_karatsuba(f, g)

    fr = [a[0] for a in f]
    fi = [a[1] for a in f]
    gr = [a[0] for a in g]
    gi = [a[1] for a in g]

    hr = [a - b for a, b in zip(karatsuba(fr, gr), karatsuba(fi, gi))]
    hi = [a + b for a, b in zip(karatsuba(fr, gi), karatsuba(fi, gr))]
    h_ = [(a, b) for a, b in zip(hr, hi)]

    assert(h_ == h), f"complex karatsuba not working\ngot: {h}\nexp: {h_}"

    print(h)

def complex_negacyclomul(f, g):
    n = len(f)
    half = n//2

    flo = f[:half]
    fhi = f[half:]

    glo = g[:half]
    ghi = g[half:]

    f0 = [(lo, -hi) for lo, hi in zip(flo, fhi)]
    f1 = [(lo, hi) for lo, hi in zip(flo, fhi)]
    g0 = [(lo, -hi) for lo, hi in zip(glo, ghi)]
    g1 = [(lo, hi) for lo, hi in zip(glo, ghi)]

    h0 = complex_karatsuba(f0, g0)

    #h1 = complex_karatsuba(f1, g1)

    # h = a * h0 + b * h1
    # where a = 2^-1 * (i*X^(n/2) + 1)
    # and  b = 2^-1 * (-i*X^(n/2) + 1)

    field = f[0].parent()
    h = [field(0) for i in range(2*n-1)]
    #hi = [field(0) for i in range(2*n-1)]
    for i in range(len(h0)):
        h[i] += h0[i][0] # / 2
        h[i+half] -= h0[i][1] # / 2
        #h[i] += h1[i][0] / 2
        #h[i+half] += h1[i][1] / 2

    hh = [field(0) for i in range(n)]
    for i in range(len(h)):
        if i < n:
            hh[i] += h[i]
        else:
            hh[i-n] -= h[i]

    return hh

def test_complex_negacyclomul():
    p = 2^64 - 2^32 + 1
    field = FiniteField(p)

    f = [field.random_element() for i in range(8)]
    g = [field.random_element() for i in range(8)]

    f = [ZZ(i) for i in range(8)]
    g = [ZZ(1) for i in range(8)]

    h = negacyclomul(f,g)

    h_ = complex_negacyclomul(f,g)

    assert(h_ == h), f"complex negacyclomul does not give same result as ordinary negacyclomul.\ngot: {h_}\nexp: {h}"


