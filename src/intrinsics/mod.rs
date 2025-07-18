//! Platform-specific intrinsics for AES operations with compile-time dispatch.

/// AES round function without key addition: MixColumns(ShiftRows(SubBytes(x))).
///
/// This function applies a single AES encryption round without the AddRoundKey step.
/// The implementation is selected at compile-time based on target features.
#[inline]
pub fn aesl(block: &[u8; 16]) -> [u8; 16] {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    {
        aarch64::aesl(block)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
    {
        x86_64::aesl(block)
    }
    #[cfg(not(any(
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        ),
        all(target_arch = "x86_64", target_feature = "aes")
    )))]
    {
        fallback::aesl(block)
    }
}

/// SIMD XOR operation for 16-byte blocks.
#[inline]
pub fn xor_block_simd(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::xor_block(a, b)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        x86_64::xor_block(a, b)
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        // Fallback XOR for 16-byte blocks
        let mut result = [0u8; 16];
        for i in 0..16 {
            result[i] = a[i] ^ b[i];
        }
        result
    }
}

/// SIMD reduction XOR for multiple blocks.
#[inline]
pub fn xor_reduce_blocks(blocks: &[[u8; 16]; 16]) -> [u8; 16] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::xor_reduce_blocks(blocks)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        x86_64::xor_reduce_blocks(blocks)
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        fallback::xor_reduce_blocks(blocks)
    }
}

/// Platform-optimized fused AESL+XOR operations.
/// These functions provide architecture-specific optimizations for common HiAE patterns.
/// ARM NEON optimized XAESL: Computes AESL(x^y) in a single fused operation.
#[inline]
pub fn xaesl(x: &[u8; 16], y: &[u8; 16]) -> [u8; 16] {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    {
        aarch64::xaesl(x, y)
    }
    #[cfg(not(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    )))]
    {
        // Fallback: separate XOR and AESL operations
        let xor_result = xor_block_simd(x, y);
        aesl(&xor_result)
    }
}

/// ARM NEON optimized three-way XOR: Computes x^y^z.
#[allow(dead_code)]
#[inline]
pub fn xor3(x: &[u8; 16], y: &[u8; 16], z: &[u8; 16]) -> [u8; 16] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::xor3(x, y, z)
    }
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        // Fallback: two separate XOR operations
        let temp = xor_block_simd(x, y);
        xor_block_simd(&temp, z)
    }
}

/// Intel AES-NI optimized AESLX: Computes AESL(y) ^ z using native instructions.
#[allow(dead_code)]
#[inline]
pub fn aeslx(y: &[u8; 16], z: &[u8; 16]) -> [u8; 16] {
    #[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
    {
        x86_64::aeslx(y, z)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "aes")))]
    {
        // Fallback: separate AESL and XOR operations
        let aesl_result = aesl(y);
        xor_block_simd(&aesl_result, z)
    }
}

/// Platform-optimized batch processing functions.
/// These functions provide high-throughput operations for multiple blocks.
/// Batch AESL processing for 4 blocks simultaneously.
/// Uses platform-specific SIMD operations to maximize throughput.
#[allow(dead_code)]
#[inline]
pub fn aesl_batch4(blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    #[cfg(all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ))]
    {
        aarch64::aesl_batch4(blocks)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
    {
        x86_64::aesl_batch4(blocks)
    }
    #[cfg(not(any(
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        ),
        all(target_arch = "x86_64", target_feature = "aes")
    )))]
    {
        fallback::aesl_batch4(blocks)
    }
}

/// Batch XOR processing for 4 block pairs simultaneously.
/// Uses platform-specific SIMD operations to maximize throughput.
#[allow(dead_code)]
#[inline]
pub fn xor_batch4(a_blocks: &[[u8; 16]; 4], b_blocks: &[[u8; 16]; 4]) -> [[u8; 16]; 4] {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        aarch64::xor_batch4(a_blocks, b_blocks)
    }
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        x86_64::xor_batch4(a_blocks, b_blocks)
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "sse2")
    )))]
    {
        fallback::xor_batch4(a_blocks, b_blocks)
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

mod fallback;
