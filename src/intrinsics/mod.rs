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

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

mod fallback;
