//! Platform-specific intrinsics for AES operations with compile-time dispatch.

/// AES round function without key addition: MixColumns(ShiftRows(SubBytes(x))).
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
#[allow(dead_code)]
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
        let xor_result = xor_block_simd(x, y);
        aesl(&xor_result)
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
        let aesl_result = aesl(y);
        xor_block_simd(&aesl_result, z)
    }
}

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

mod fallback;
