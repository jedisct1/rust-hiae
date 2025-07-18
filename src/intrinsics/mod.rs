//! Platform-specific intrinsics for AES operations.

use core::sync::atomic::{AtomicU8, Ordering};

/// CPU capability detection cache.
static CPU_FEATURES: AtomicU8 = AtomicU8::new(0);

/// Feature detection states.
const FEATURES_UNKNOWN: u8 = 0;
const FEATURES_HARDWARE: u8 = 1;
const FEATURES_SOFTWARE: u8 = 2;

/// Initialize CPU feature detection (called once).
#[cold]
fn init_cpu_features() -> u8 {
    #[cfg(target_arch = "aarch64")]
    {
        #[cfg(feature = "std")]
        {
            if std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("aes")
            {
                return FEATURES_HARDWARE;
            }
        }
        #[cfg(not(feature = "std"))]
        {
            // In no-std mode, compile-time detection only
            #[cfg(all(target_feature = "neon", target_feature = "aes"))]
            {
                return FEATURES_HARDWARE;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "std")]
        {
            if std::arch::is_x86_feature_detected!("aes") {
                return FEATURES_HARDWARE;
            }
        }
        #[cfg(not(feature = "std"))]
        {
            #[cfg(target_feature = "aes")]
            {
                return FEATURES_HARDWARE;
            }
        }
    }

    FEATURES_SOFTWARE
}

/// Get CPU features with caching.
#[inline]
fn get_cpu_features() -> u8 {
    let features = CPU_FEATURES.load(Ordering::Relaxed);
    if features == FEATURES_UNKNOWN {
        let detected = init_cpu_features();
        CPU_FEATURES.store(detected, Ordering::Relaxed);
        detected
    } else {
        features
    }
}

/// AES round function without key addition: MixColumns(ShiftRows(SubBytes(x))).
///
/// This function applies a single AES encryption round without the AddRoundKey step.
/// The implementation is selected at runtime based on the CPU's actual capabilities.
#[inline]
pub fn aesl(block: &[u8; 16]) -> [u8; 16] {
    if get_cpu_features() == FEATURES_HARDWARE {
        #[cfg(target_arch = "aarch64")]
        {
            return aarch64::aesl(block);
        }
        #[cfg(target_arch = "x86_64")]
        {
            return x86_64::aesl(block);
        }
    }

    // Fallback to portable implementation
    fallback::aesl(block)
}

/// SIMD XOR operation for 16-byte blocks.
#[inline]
pub fn xor_block_simd(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    if get_cpu_features() == FEATURES_HARDWARE {
        #[cfg(target_arch = "aarch64")]
        {
            return aarch64::xor_block(a, b);
        }
        #[cfg(target_arch = "x86_64")]
        {
            return x86_64::xor_block(a, b);
        }
    }

    // Fallback to portable implementation
    fallback::xor_block(a, b)
}

/// SIMD reduction XOR for multiple blocks.
#[inline]
pub fn xor_reduce_blocks(blocks: &[[u8; 16]; 16]) -> [u8; 16] {
    if get_cpu_features() == FEATURES_HARDWARE {
        #[cfg(target_arch = "aarch64")]
        {
            return aarch64::xor_reduce_blocks(blocks);
        }
        #[cfg(target_arch = "x86_64")]
        {
            return x86_64::xor_reduce_blocks(blocks);
        }
    }

    // Fallback to portable implementation
    fallback::xor_reduce_blocks(blocks)
}

#[cfg(target_arch = "aarch64")]
mod aarch64;

#[cfg(target_arch = "x86_64")]
mod x86_64;

mod fallback;
