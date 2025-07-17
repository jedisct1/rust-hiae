//! Platform-specific intrinsics for AES operations.

/// AES round function without key addition: MixColumns(ShiftRows(SubBytes(x))).
///
/// This function applies a single AES encryption round without the AddRoundKey step.
/// The implementation is selected at compile time based on the target architecture
/// and available features.
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

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
mod aarch64;

#[cfg(all(target_arch = "x86_64", target_feature = "aes"))]
mod x86_64;

mod fallback;
