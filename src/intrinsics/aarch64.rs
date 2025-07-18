//! ARM NEON + Crypto Extensions optimized implementation.

#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

/// ARM NEON + Crypto optimized AESL implementation.
///
/// This implementation leverages the ARM Crypto Extensions which provide
/// dedicated AES instructions. The pattern z ^ AESL(x) is optimized using
/// fused NEON operations.
#[target_feature(enable = "neon,aes")]
unsafe fn aesl_impl(block: &[u8; 16]) -> [u8; 16] {
    // Load the input block into a NEON register
    let input = vld1q_u8(block.as_ptr());

    // Zero key for AES round (since we want AESL without AddRoundKey)
    let zero_key = vdupq_n_u8(0);

    // Apply AES encryption round: SubBytes + ShiftRows + MixColumns
    // vaeseq_u8 performs SubBytes + ShiftRows + AddRoundKey
    // Since we want AESL without AddRoundKey, we use zero as the key
    let after_sub_shift = vaeseq_u8(input, zero_key);

    // vaesmcq_u8 performs MixColumns
    let result = vaesmcq_u8(after_sub_shift);

    // Store result back to array
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
    // Load both blocks into NEON registers
    let a_vec = vld1q_u8(a.as_ptr());
    let b_vec = vld1q_u8(b.as_ptr());

    // Perform vectorized XOR
    let result_vec = veorq_u8(a_vec, b_vec);

    // Store result back to array
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
    // Load first block
    let mut result = vld1q_u8(blocks[0].as_ptr());

    // XOR all remaining blocks
    for block in blocks.iter().skip(1) {
        let block_vec = vld1q_u8(block.as_ptr());
        result = veorq_u8(result, block_vec);
    }

    // Store result
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
    // Load both input blocks into NEON registers
    let x_vec = vld1q_u8(x.as_ptr());
    let y_vec = vld1q_u8(y.as_ptr());

    // Compute x^y using NEON XOR
    let xor_result = veorq_u8(x_vec, y_vec);

    // Apply AESL: vaeseq_u8 followed by vaesmcq_u8
    // This is equivalent to AESL(x^y) in a single fused operation
    let after_sub_shift = vaeseq_u8(xor_result, vdupq_n_u8(0));
    let result = vaesmcq_u8(after_sub_shift);

    // Store result back to array
    let mut output = [0u8; 16];
    vst1q_u8(output.as_mut_ptr(), result);
    output
}

/// Safe wrapper for ARM-optimized XAESL function.
#[inline]
pub fn xaesl(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    unsafe { xaesl_impl(x, y) }
}

/// ARM-optimized three-way XOR function using EOR3 instruction when available.
/// Computes x^y^z in a single instruction on ARMv8.2+ with SHA3 extensions.
#[target_feature(enable = "neon")]
#[allow(dead_code)]
unsafe fn xor3_impl(x: &[u8; 16], y: &[u8; 16], z: &[u8; 16]) -> [u8; 16] {
    // Load all three input blocks into NEON registers
    let x_vec = vld1q_u8(x.as_ptr());
    let y_vec = vld1q_u8(y.as_ptr());
    let z_vec = vld1q_u8(z.as_ptr());

    // Check if EOR3 is available (ARM SHA3 extensions)
    #[cfg(target_feature = "sha3")]
    {
        // Use single EOR3 instruction for three-way XOR
        let result = veor3q_u8(x_vec, y_vec, z_vec);
        let mut output = [0u8; 16];
        vst1q_u8(output.as_mut_ptr(), result);
        output
    }
    #[cfg(not(target_feature = "sha3"))]
    {
        // Fallback to two XOR operations
        let temp = veorq_u8(x_vec, y_vec);
        let result = veorq_u8(temp, z_vec);
        let mut output = [0u8; 16];
        vst1q_u8(output.as_mut_ptr(), result);
        output
    }
}

/// Safe wrapper for ARM-optimized three-way XOR function.
#[inline]
#[allow(dead_code)]
pub fn xor3(x: &[u8; 16], y: &[u8; 16], z: &[u8; 16]) -> [u8; 16] {
    unsafe { xor3_impl(x, y, z) }
}

/// ARM-optimized batch AESL processing for multiple blocks in parallel.
/// Processes 4 blocks simultaneously using multiple NEON registers.
#[target_feature(enable = "neon,aes")]
#[allow(dead_code)]
unsafe fn aesl_batch4_impl(blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    // Load all 4 blocks into NEON registers
    let block0 = vld1q_u8(blocks[0].as_ptr());
    let block1 = vld1q_u8(blocks[1].as_ptr());
    let block2 = vld1q_u8(blocks[2].as_ptr());
    let block3 = vld1q_u8(blocks[3].as_ptr());

    // Zero key for AES rounds
    let zero_key = vdupq_n_u8(0);

    // Apply AESL to all blocks in parallel
    let result0 = vaesmcq_u8(vaeseq_u8(block0, zero_key));
    let result1 = vaesmcq_u8(vaeseq_u8(block1, zero_key));
    let result2 = vaesmcq_u8(vaeseq_u8(block2, zero_key));
    let result3 = vaesmcq_u8(vaeseq_u8(block3, zero_key));

    // Store results
    let mut output = [[0u8; 16]; 4];
    vst1q_u8(output[0].as_mut_ptr(), result0);
    vst1q_u8(output[1].as_mut_ptr(), result1);
    vst1q_u8(output[2].as_mut_ptr(), result2);
    vst1q_u8(output[3].as_mut_ptr(), result3);
    output
}

/// Safe wrapper for ARM-optimized batch AESL processing.
#[inline]
#[allow(dead_code)]
pub fn aesl_batch4(blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    unsafe { aesl_batch4_impl(blocks) }
}

/// ARM-optimized batch XOR processing for multiple block pairs.
/// Processes 4 block pairs simultaneously using multiple NEON registers.
#[target_feature(enable = "neon")]
#[allow(dead_code)]
unsafe fn xor_batch4_impl(a_blocks: &[[u8; 16]; 4], b_blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    // Load all blocks into NEON registers
    let a0 = vld1q_u8(a_blocks[0].as_ptr());
    let a1 = vld1q_u8(a_blocks[1].as_ptr());
    let a2 = vld1q_u8(a_blocks[2].as_ptr());
    let a3 = vld1q_u8(a_blocks[3].as_ptr());

    let b0 = vld1q_u8(b_blocks[0].as_ptr());
    let b1 = vld1q_u8(b_blocks[1].as_ptr());
    let b2 = vld1q_u8(b_blocks[2].as_ptr());
    let b3 = vld1q_u8(b_blocks[3].as_ptr());

    // Perform vectorized XOR operations in parallel
    let result0 = veorq_u8(a0, b0);
    let result1 = veorq_u8(a1, b1);
    let result2 = veorq_u8(a2, b2);
    let result3 = veorq_u8(a3, b3);

    // Store results
    let mut output = [[0u8; 16]; 4];
    vst1q_u8(output[0].as_mut_ptr(), result0);
    vst1q_u8(output[1].as_mut_ptr(), result1);
    vst1q_u8(output[2].as_mut_ptr(), result2);
    vst1q_u8(output[3].as_mut_ptr(), result3);
    output
}

/// Safe wrapper for ARM-optimized batch XOR processing.
#[inline]
#[allow(dead_code)]
pub fn xor_batch4(a_blocks: &[[u8; 16]; 4], b_blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    unsafe { xor_batch4_impl(a_blocks, b_blocks) }
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

    #[test]
    fn test_arm_xor3() {
        let x = [0x11; 16];
        let y = [0x22; 16];
        let z = [0x44; 16];

        // XOR3(x, y, z) should equal x XOR y XOR z
        let mut manual_result = [0u8; 16];
        for i in 0..16 {
            manual_result[i] = x[i] ^ y[i] ^ z[i];
        }

        let xor3_result = unsafe { xor3_impl(&x, &y, &z) };
        assert_eq!(xor3_result, manual_result);
    }
}
