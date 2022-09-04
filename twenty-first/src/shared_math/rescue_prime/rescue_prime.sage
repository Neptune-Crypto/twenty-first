# rescue_prime.sage
# The reference implementation of regular Rescue-Prime for Neptune.

from CompactFIPS202 import SHAKE256

def get_round_constants( p, m, capacity, security_level, N ):
    # generate pseudorandom bytes
    bytes_per_int = ceil(len(bin(p)[2:]) / 8) + 1
    num_bytes = bytes_per_int * 2 * m * N
    seed_string = "Rescue-XLIX(%i,%i,%i,%i)" % (p, m, capacity, security_level)
    byte_string = SHAKE256(bytes(seed_string, "ascii"), num_bytes)

    # process byte string in chunks
    round_constants = []
    Fp = FiniteField(p)
    for i in range(2*m*N):
        chunk = byte_string[bytes_per_int*i : bytes_per_int*(i+1)]
        integer = sum(256^j * ZZ(chunk[j]) for j in range(len(chunk)))
        round_constants.append(Fp(integer % p))

    return round_constants

def get_alphas( p ):
    for alpha in range(3, p):
        if gcd(alpha, p-1) == 1:
            break
    g, alphainv, garbage = xgcd(alpha, p-1)
    return (alpha, (alphainv % (p-1)))

def get_mds_matrix( p, m ):
    # get a primitive element
    Fp = FiniteField(p)
    g = Fp(2)
    while g.multiplicative_order() != p-1:
        g = g + 1

    # get a systematic generator matrix for the code
    V = matrix([[g^(i*j) for j in range(0, 2*m)] for i in range(0, m)])
    V_ech = V.echelon_form()

    # the MDS matrix is the transpose of the right half of this matrix
    MDS = V_ech[:, m:].transpose()
    return MDS

def rescue_prime_hash_varlen( parameters, input_sequence ):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    rate = m - capacity
    Fp = FiniteField(p)

    # pad
    padded_input = input_sequence + [Fp(1)]
    while len(padded_input) % rate != 0:
        padded_input.append(Fp(0))

    # divide into chunks of 6 and apply merkle-damgaard
    chaining_value = [Fp(0)] * 6
    while padded_input:
        chaining_value = rescue_prime_hash_12(parameters, chaining_value + padded_input[:6])
        padded_input = padded_input[6:]

    return chaining_value

def rescue_prime_hash_12( parameters, input_sequence ):

    assert len(input_sequence) == 12, "Function can only hash sequences of 12 field elements; try `rescue_prime_hash_varlen` instead."

    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    rate = m - capacity
    Fp = FiniteField(p)

    # initialize state to all zeros
    state = matrix([inp for inp in input_sequence] + [Fp(0) for i in range(capacity)]).transpose()

    # apply permutation
    state = rescue_XLIX_permutation(parameters, state)

    # get top 6 elements
    output_sequence = [state[i,0] for i in range(6)]

    return output_sequence

def rescue_XLIX_permutation( parameters, state ):
    p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants = parameters
    Fp = state[0,0].parent()

    for i in range(N):
        # S-box
        for j in range(m):
            state[j,0] = state[j,0]^alpha
        # mds
        state = MDS * state
        # constants
        for j in range(m):
            state[j,0] += round_constants[i*2*m+j]

        # inverse S-box
        for j in range(m):
            state[j,0] = state[j,0]^alphainv
        # mds
        state = MDS * state
        # constants
        for j in range(m):
            state[j,0] += round_constants[i*2*m+m+j]

    return state

def print_test_vectors():
    Fp = FiniteField(p)

    print("Hash 12:")
    input_sequence = [Fp(0)] * 12
    for i in range(10):
        input_sequence[-1] = Fp(i)
        output_sequence = rescue_prime_hash_12(parameters, input_sequence)
        print(matrix([input_sequence]), " -> ", matrix([output_sequence]))
    input_sequence[-1] = Fp(0)
    for i in range(12):
        input_sequence[i] = Fp(1)
        output_sequence = rescue_prime_hash_12(parameters, input_sequence)
        print(matrix([input_sequence]), " -> ", matrix([output_sequence]))
        input_sequence[i] = Fp(0)

    print("Hash Variable Length:")
    for i in range(20):
        input_sequence = [Fp(j) for j in range(i)]
        output_sequence = rescue_prime_hash_varlen(parameters, input_sequence)
        print(matrix([input_sequence]), " -> ", matrix([output_sequence]))

p = 2^64 - 2^32 + 1
m = 16
rate = 12
capacity = 4
security_level = 160
N = 8
alpha, alphainv = get_alphas(p)
MDS = get_mds_matrix(p, m)
round_constants = get_round_constants( p, m, capacity, security_level, N )

parameters = (p, m, capacity, security_level, alpha, alphainv, N, MDS, round_constants)


