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

/// SIMD XOR implementation for x86-64 SSE2.
#[target_feature(enable = "sse2")]
unsafe fn xor_block_impl(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    // Load both blocks into SSE registers
    let a_vec = _mm_loadu_si128(a.as_ptr() as *const __m128i);
    let b_vec = _mm_loadu_si128(b.as_ptr() as *const __m128i);

    // Perform vectorized XOR
    let result_vec = _mm_xor_si128(a_vec, b_vec);

    // Store result back to array
    let mut result = [0u8; 16];
    _mm_storeu_si128(result.as_mut_ptr() as *mut __m128i, result_vec);
    result
}

/// Safe wrapper around the x86-64 SSE2 XOR implementation.
#[inline]
pub fn xor_block(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    unsafe { xor_block_impl(a, b) }
}

/// SIMD reduction XOR for 16 blocks using x86-64 SSE2.
#[target_feature(enable = "sse2")]
unsafe fn xor_reduce_blocks_impl(blocks: &[[u8; 16]; 16]) -> [u8; 16] {
    // Load first block
    let mut result = _mm_loadu_si128(blocks[0].as_ptr() as *const __m128i);

    // XOR all remaining blocks
    for block in blocks.iter().skip(1) {
        let block_vec = _mm_loadu_si128(block.as_ptr() as *const __m128i);
        result = _mm_xor_si128(result, block_vec);
    }

    // Store result
    let mut output = [0u8; 16];
    _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
    output
}

/// Safe wrapper around the x86-64 SSE2 XOR reduction implementation.
#[inline]
pub fn xor_reduce_blocks(blocks: &[[u8; 16]; 16]) -> [u8; 16] {
    unsafe { xor_reduce_blocks_impl(blocks) }
}

/// Intel-optimized AESLX function: Computes AESL(y) ^ z using native AES-NI.
/// This leverages the x86-64 AES-NI _mm_aesenc_si128 instruction for efficiency.
#[target_feature(enable = "aes")]
unsafe fn aeslx_impl(y: &[u8; 16], z: &[u8; 16]) -> [u8; 16] {
    // Load both input blocks into SSE registers
    let y_vec = _mm_loadu_si128(y.as_ptr() as *const __m128i);
    let z_vec = _mm_loadu_si128(z.as_ptr() as *const __m128i);

    // Apply AES encryption round using z as the round key
    // _mm_aesenc_si128(y, z) computes AESL(y) ^ z in a single instruction
    let result = _mm_aesenc_si128(y_vec, z_vec);

    // Store result back to array
    let mut output = [0u8; 16];
    _mm_storeu_si128(output.as_mut_ptr() as *mut __m128i, result);
    output
}

/// Safe wrapper for Intel-optimized AESLX function.
#[inline]
pub fn aeslx(y: &[u8; 16], z: &[u8; 16]) -> [u8; 16] {
    unsafe { aeslx_impl(y, z) }
}

/// Intel-optimized batch AESL processing with instruction pipelining.
/// Processes 4 blocks simultaneously to hide instruction latency.
#[target_feature(enable = "aes")]
unsafe fn aesl_batch4_impl(blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    // Load all 4 blocks into SSE registers
    let block0 = _mm_loadu_si128(blocks[0].as_ptr() as *const __m128i);
    let block1 = _mm_loadu_si128(blocks[1].as_ptr() as *const __m128i);
    let block2 = _mm_loadu_si128(blocks[2].as_ptr() as *const __m128i);
    let block3 = _mm_loadu_si128(blocks[3].as_ptr() as *const __m128i);

    // Zero key for AES rounds
    let zero_key = _mm_setzero_si128();

    // Apply AESL to all blocks in parallel with instruction interleaving
    // This helps hide the latency of AES instructions
    let result0 = _mm_aesenc_si128(block0, zero_key);
    let result1 = _mm_aesenc_si128(block1, zero_key);
    let result2 = _mm_aesenc_si128(block2, zero_key);
    let result3 = _mm_aesenc_si128(block3, zero_key);

    // Store results
    let mut output = [[0u8; 16]; 4];
    _mm_storeu_si128(output[0].as_mut_ptr() as *mut __m128i, result0);
    _mm_storeu_si128(output[1].as_mut_ptr() as *mut __m128i, result1);
    _mm_storeu_si128(output[2].as_mut_ptr() as *mut __m128i, result2);
    _mm_storeu_si128(output[3].as_mut_ptr() as *mut __m128i, result3);
    output
}

/// Safe wrapper for Intel-optimized batch AESL processing.
#[inline]
pub fn aesl_batch4(blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    unsafe { aesl_batch4_impl(blocks) }
}

/// Intel-optimized batch XOR processing with better instruction scheduling.
/// Processes 4 block pairs simultaneously using multiple SSE registers.
#[target_feature(enable = "sse2")]
unsafe fn xor_batch4_impl(a_blocks: &[[u8; 16]; 4], b_blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    // Load all blocks into SSE registers
    let a0 = _mm_loadu_si128(a_blocks[0].as_ptr() as *const __m128i);
    let a1 = _mm_loadu_si128(a_blocks[1].as_ptr() as *const __m128i);
    let a2 = _mm_loadu_si128(a_blocks[2].as_ptr() as *const __m128i);
    let a3 = _mm_loadu_si128(a_blocks[3].as_ptr() as *const __m128i);

    let b0 = _mm_loadu_si128(b_blocks[0].as_ptr() as *const __m128i);
    let b1 = _mm_loadu_si128(b_blocks[1].as_ptr() as *const __m128i);
    let b2 = _mm_loadu_si128(b_blocks[2].as_ptr() as *const __m128i);
    let b3 = _mm_loadu_si128(b_blocks[3].as_ptr() as *const __m128i);

    // Perform vectorized XOR operations in parallel
    let result0 = _mm_xor_si128(a0, b0);
    let result1 = _mm_xor_si128(a1, b1);
    let result2 = _mm_xor_si128(a2, b2);
    let result3 = _mm_xor_si128(a3, b3);

    // Store results
    let mut output = [[0u8; 16]; 4];
    _mm_storeu_si128(output[0].as_mut_ptr() as *mut __m128i, result0);
    _mm_storeu_si128(output[1].as_mut_ptr() as *mut __m128i, result1);
    _mm_storeu_si128(output[2].as_mut_ptr() as *mut __m128i, result2);
    _mm_storeu_si128(output[3].as_mut_ptr() as *mut __m128i, result3);
    output
}

/// Safe wrapper for Intel-optimized batch XOR processing.
#[inline]
pub fn xor_batch4(a_blocks: &[[u8; 16]; 4], b_blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    unsafe { xor_batch4_impl(a_blocks, b_blocks) }
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

    #[test]
    fn test_x86_aeslx() {
        let y = [0x11; 16];
        let z = [0x22; 16];

        // AESLX(y, z) should equal AESL(y) XOR z
        let manual_aesl = unsafe { aesl_impl(&y) };
        let manual_result = super::super::xor_block_simd(&manual_aesl, &z);
        let aeslx_result = unsafe { aeslx_impl(&y, &z) };

        assert_eq!(aeslx_result, manual_result);
    }
}
