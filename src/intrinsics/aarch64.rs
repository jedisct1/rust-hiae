//! ARM NEON + Crypto Extensions optimized implementation.

#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// ARM NEON + Crypto optimized AESL implementation.
/// This implementation leverages the ARM Crypto Extensions which provide
/// dedicated AES instructions. The pattern z ^ AESL(x) is optimized using
/// fused NEON operations.
#[target_feature(enable = "neon,aes")]
unsafe fn aesl_impl(block: &[u8; 16]) -> [u8; 16] {
    let input = vld1q_u8(block.as_ptr());

    // Zero key for AES round (since we want AESL without AddRoundKey)
    let zero_key = vdupq_n_u8(0);

    // Apply AES encryption round: SubBytes + ShiftRows + MixColumns
    // vaeseq_u8 performs SubBytes + ShiftRows + AddRoundKey
    // Since we want AESL without AddRoundKey, we use zero as the key
    let after_sub_shift = vaeseq_u8(input, zero_key);

    // vaesmcq_u8 performs MixColumns
    let result = vaesmcq_u8(after_sub_shift);

    let mut output = [0u8; 16];
    vst1q_u8(output.as_mut_ptr(), result);
    output
}

/// Safe wrapper around the ARM implementation.
#[inline]
pub fn aesl(block: &[u8; 16]) -> [u8; 16] {
    unsafe { aesl_impl(block) }
}

/// SIMD XOR implementation for ARM NEON.
#[target_feature(enable = "neon")]
unsafe fn xor_block_impl(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let a_vec = vld1q_u8(a.as_ptr());
    let b_vec = vld1q_u8(b.as_ptr());

    // Perform vectorized XOR
    let result_vec = veorq_u8(a_vec, b_vec);

    let mut result = [0u8; 16];
    vst1q_u8(result.as_mut_ptr(), result_vec);
    result
}

/// Safe wrapper around the ARM NEON XOR implementation.
#[inline]
pub fn xor_block(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    unsafe { xor_block_impl(a, b) }
}

/// SIMD reduction XOR for 16 blocks using ARM NEON.
#[target_feature(enable = "neon")]
unsafe fn xor_reduce_blocks_impl(blocks: &[[u8; 16]; 16]) -> [u8; 16] {
    let mut result = vld1q_u8(blocks[0].as_ptr());

    // XOR all remaining blocks
    for block in blocks.iter().skip(1) {
        let block_vec = vld1q_u8(block.as_ptr());
        result = veorq_u8(result, block_vec);
    }

    let mut output = [0u8; 16];
    vst1q_u8(output.as_mut_ptr(), result);
    output
}

/// Safe wrapper around the ARM NEON XOR reduction implementation.
#[inline]
pub fn xor_reduce_blocks(blocks: &[[u8; 16]; 16]) -> [u8; 16] {
    unsafe { xor_reduce_blocks_impl(blocks) }
}

/// ARM-optimized XAESL function: Computes AESL(x^y) in a single fused operation.
/// This leverages the ARM Crypto Extensions for maximum efficiency.
#[target_feature(enable = "neon,aes")]
unsafe fn xaesl_impl(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    let x_vec = vld1q_u8(x.as_ptr());
    let y_vec = vld1q_u8(y.as_ptr());

    // Compute x^y using NEON XOR
    let xor_result = veorq_u8(x_vec, y_vec);

    // Apply AESL: vaeseq_u8 followed by vaesmcq_u8
    // This is equivalent to AESL(x^y) in a single fused operation
    let after_sub_shift = vaeseq_u8(xor_result, vdupq_n_u8(0));
    let result = vaesmcq_u8(after_sub_shift);

    let mut output = [0u8; 16];
    vst1q_u8(output.as_mut_ptr(), result);
    output
}

/// Safe wrapper for ARM-optimized XAESL function.
#[inline]
pub fn xaesl(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    unsafe { xaesl_impl(x, y) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arm_aesl() {
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
    fn test_arm_xaesl() {
        let x = [0x11; 16];
        let y = [0x22; 16];

        // XAESL(x, y) should equal AESL(x XOR y)
        let manual_xor = super::super::xor_block_simd(&x, &y);
        let manual_result = unsafe { aesl_impl(&manual_xor) };
        let xaesl_result = unsafe { xaesl_impl(&x, &y) };

        assert_eq!(xaesl_result, manual_result);
    }
}
