//! Platform-specific intrinsics for AES operations.

/// AES round function without key addition: MixColumns(ShiftRows(SubBytes(x))).
///
/// This function applies a single AES encryption round without the AddRoundKey step.
/// The implementation is selected at runtime based on the CPU's actual capabilities.
#[inline]
pub fn aesl(block: &[u8; 16]) -> [u8; 16] {
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(feature = "std")]
        {
            if std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("aes")
            {
                return aarch64::aesl(block);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            // In no-std mode, compile-time detection only
            #[cfg(all(target_feature = "neon", target_feature = "aes"))]
            {
                return aarch64::aesl(block);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::arch::is_x86_feature_detected!("aes") {
                return x86_64::aesl(block);
            }
        }
        #[cfg(not(feature = "std"))]
        {
            #[cfg(target_feature = "aes")]
            {
                return x86_64::aesl(block);
            }
        }
    }

    // Fallback to portable implementation
    fallback::aesl(block)
}

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

mod fallback;
