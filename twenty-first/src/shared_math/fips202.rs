fn state_bytes_to_u64s(bytes: &[u8; 200]) -> [u64; 25] {
    let mut u64s = [0u64; 25];
    for i in 0..25 {
        u64s[i] = u64::from_le_bytes(bytes[8 * i..8 * (i + 1)].try_into().unwrap());
    }
    u64s
}

fn state_u64s_to_bytes(u64s: &[u64; 25]) -> [u8; 200] {
    let mut bytes = [0u8; 200];
    for i in 0..25 {
        let integer_bytes = u64::to_le_bytes(u64s[i]);
        for j in 0..8 {
            bytes[i * 8 + j] = integer_bytes[j];
        }
    }
    bytes
}

fn keccak(
    rate: usize,
    capacity: usize,
    input: &[u8],
    delimiter_suffix: u8,
    output_length: usize,
) -> Vec<u8> {
    let mut state_bytes = [0u8; 200];
    let mut state_u64s: [u64; 25];
    let rate_in_bytes = rate / 8;
    let mut block_size = 0;

    debug_assert_eq!(rate + capacity, 1600);
    debug_assert_eq!(rate % 8, 0);

    /* === Absorb all the input blocks === */
    let mut num_bytes_remaining = input.len();
    let mut input_offset = 0;
    while num_bytes_remaining > 0 {
        block_size = if num_bytes_remaining < rate_in_bytes {
            num_bytes_remaining
        } else {
            rate_in_bytes
        };
        for i in 0..block_size {
            state_bytes[i] ^= input[input_offset + i];
        }
        input_offset += block_size;
        num_bytes_remaining -= block_size;

        if block_size == rate_in_bytes {
            state_u64s = state_bytes_to_u64s(&state_bytes);
            keccak::f1600(&mut state_u64s);
            state_bytes = state_u64s_to_bytes(&state_u64s);
            block_size = 0;
        }
    }

    /* === Do the padding and switch to the squeezing phase === */
    /* Absorb the last few bits and add the first bit of padding (which coincides with the delimiter in delimitedSuffix) */
    state_bytes[block_size] ^= delimiter_suffix;
    /* If the first bit of padding is at position rate-1, we need a whole new block for the second bit of padding */
    if ((delimiter_suffix & 0x80) != 0) && (block_size == (rate_in_bytes - 1)) {
        state_u64s = state_bytes_to_u64s(&state_bytes);
        keccak::f1600(&mut state_u64s);
        state_bytes = state_u64s_to_bytes(&state_u64s);
    }
    /* Add the second bit of padding */
    state_bytes[rate_in_bytes - 1] ^= 0x80;
    /* Switch to the squeezing phase */
    state_u64s = state_bytes_to_u64s(&state_bytes);
    keccak::f1600(&mut state_u64s);
    state_bytes = state_u64s_to_bytes(&state_u64s);

    /* === Squeeze out all the output blocks === */
    let mut output: Vec<u8> = Vec::with_capacity(output_length);
    num_bytes_remaining = output_length;
    while num_bytes_remaining > 0 {
        block_size = if num_bytes_remaining < rate_in_bytes {
            num_bytes_remaining
        } else {
            rate_in_bytes
        };
        // output.append(&mut state_bytes[0..block_size].to_owned());
        for byte in state_bytes.iter().take(block_size) {
            output.push(*byte);
        }
        num_bytes_remaining -= block_size;

        if num_bytes_remaining > 0 {
            state_u64s = state_bytes_to_u64s(&state_bytes);
            keccak::f1600(&mut state_u64s);
            state_bytes = state_u64s_to_bytes(&state_u64s);
        }
    }

    output
}

/// Function to compute SHAKE256 on the input message with any output length.
pub fn shake128(input: &[u8], output_length: usize) -> Vec<u8> {
    keccak(1344, 256, input, 0x1F, output_length)
}

///  Function to compute SHAKE256 on the input message with any output length.
pub fn shake256(input: &[u8], output_length: usize) -> Vec<u8> {
    keccak(1088, 512, input, 0x1F, output_length)
}

///  Function to compute SHA3-224 on the input message. The output length is fixed to 28 bytes.
pub fn sha3_224(input: &[u8]) -> [u8; 28] {
    keccak(1152, 448, input, 0x06, 28).try_into().unwrap()
}

///  Function to compute SHA3-256 on the input message. The output length is fixed to 32 bytes.
pub fn sha3_256(input: &[u8]) -> [u8; 32] {
    keccak(1088, 512, input, 0x06, 32).try_into().unwrap()
}

///  Function to compute SHA3-384 on the input message. The output length is fixed to 48 bytes.
pub fn sha3_384(input: &[u8]) -> [u8; 48] {
    keccak(832, 768, input, 0x06, 48).try_into().unwrap()
}

///  Function to compute SHA3-512 on the input message. The output length is fixed to 64 bytes.
pub fn sha3_512(input: &[u8]) -> [u8; 64] {
    keccak(576, 1024, input, 0x06, 64).try_into().unwrap()
}

#[cfg(test)]
mod fips202_test {
    use crate::shared_math::fips202::*;

    #[test]
    fn test_kats() {
        // KATs lifted from
        // https://github.com/XKCP/XKCP/blob/master/tests/UnitTests/main.c
        // starting at line 446.
        let input = b"\x21\xF1\x34\xAC\x57";
        let output_shake128 = b"\x7B\xFB\xB4\x0D\xA3\x70\x4A\x55\x82\x91\xB3\x9E\x1E\x56\xED\x9F\x6F\x56\xAE\x78\x32\x70\xAB\x02\xA2\x02\x06\x0C\x91\x73\xFB\xB0\xB4\x55\x75\xB3\x23\x48\xA6\xED\x2C\x92\x7A\x39\xA3\x0D\xA0\xA2\xBB\xC1\x80\x74\x97\xAD\x50\xF2\x7A\x10\x77\x54\xAF\x62\x76\x2C";
        let output_shake256 = b"\xBB\x8A\x84\x47\x51\x7B\xA9\xCA\x7F\xA3\x4E\xC9\x9A\x80\x00\x4F\x22\x8A\xB2\x82\x47\x28\x41\xEB\x3D\x3A\x76\x22\x5C\x9D\xBE\x77\xF7\xE4\x0A\x06\x67\x76\xD3\x2C\x74\x94\x12\x02\xF9\xF4\xAA\x43\xD1\x2C\x62\x64\xAF\xA5\x96\x39\xC4\x4E\x11\xF5\xE1\x4F\x1E\x56";
        let output_sha3_224 = b"\x10\xE5\x80\xA3\x21\x99\x59\x61\x69\x33\x1A\xD4\x3C\xFC\xF1\x02\x64\xF8\x15\x65\x03\x70\x40\x02\x8A\x06\xB4\x58";
        let output_sha3_256 = b"\x55\xBD\x92\x24\xAF\x4E\xED\x0D\x12\x11\x49\xE3\x7F\xF4\xD7\xDD\x5B\xE2\x4B\xD9\xFB\xE5\x6E\x01\x71\xE8\x7D\xB7\xA6\xF4\xE0\x6D";
        let output_sha3_384 = b"\xE2\x48\xD6\xFF\x34\x2D\x35\xA3\x0E\xC2\x30\xBA\x51\xCD\xB1\x61\x02\x5D\x6F\x1C\x25\x1A\xCA\x6A\xE3\x53\x1F\x06\x82\xC1\x64\xA1\xFC\x07\x25\xB1\xBE\xFF\x80\x8A\x20\x0C\x13\x15\x57\xA2\x28\x09";
        let output_sha3_512 = b"\x58\x42\x19\xA8\x4E\x87\x96\x07\x6B\xF1\x17\x8B\x14\xB9\xD1\xE2\xF9\x6A\x4B\x4E\xF1\x1F\x10\xCC\x51\x6F\xBE\x1A\x29\x63\x9D\x6B\xA7\x4F\xB9\x28\x15\xF9\xE3\xC5\x19\x2E\xD4\xDC\xA2\x0A\xEA\x5B\x10\x9D\x52\x23\x7C\x99\x56\x40\x1F\xD4\x4B\x22\x1F\x82\xAB\x37";

        assert_eq!(shake128(input, 64), output_shake128.to_vec());
        assert_eq!(shake256(input, 64), output_shake256.to_vec());
        assert_eq!(sha3_224(input), *output_sha3_224);
        assert_eq!(sha3_256(input), *output_sha3_256);
        assert_eq!(sha3_384(input), *output_sha3_384);
        assert_eq!(sha3_512(input), *output_sha3_512);
    }
}
