//! x86-64 AES-NI optimized implementation.

#![allow(unsafe_code)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

/// x86-64 AES-NI optimized AESL implementation.
///
/// This implementation leverages Intel AES-NI instructions for maximum
/// performance. The pattern AESL(x) ^ y is optimized using the native
/// aesenc instruction.
#[target_feature(enable = "aes")]
unsafe fn aesl_impl(block: &[u8; 16]) -> [u8; 16] {
    // Load the input block into an SSE register
    let input = _mm_loadu_si128(block.as_ptr() as *const __m128i);

    // Zero key for AES round (since we want AESL without AddRoundKey)
    let zero_key = _mm_setzero_si128();

    // Apply AES encryption round: SubBytes + ShiftRows + MixColumns + AddRoundKey
    // Since we're adding a zero key, this gives us exactly AESL
    let result = _mm_aesenc_si128(input, zero_key);

    // Store result back to array
    let mut output = [0u8; 16];
    _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
    output
}

/// Safe wrapper around the x86-64 implementation.
#[inline]
pub fn aesl(block: &[u8; 16]) -> [u8; 16] {
    unsafe { aesl_impl(block) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_aesl() {
        let input = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let expected = [
            0x63, 0x79, 0xe6, 0xd9, 0xf4, 0x67, 0xfb, 0x76, 0xad, 0x06, 0x3c, 0xf4, 0xd2, 0xeb,
            0x8a, 0xa3,
        ];

        let result = unsafe { aesl_impl(&input) };
        assert_eq!(result, expected);
    }
}
