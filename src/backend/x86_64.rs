//! AES-NI block for x86_64.
//!
//! Only compiled when `aes` is enabled at compile time
//! (e.g. `-C target-cpu=native` or `-C target-feature=+aes`).

use super::Block;
use core::arch::x86_64::*;

#[derive(Copy, Clone)]
pub struct Aesni(__m128i);

impl Block for Aesni {
    #[inline(always)]
    fn zero() -> Self {
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    unsafe fn load(src: *const u8) -> Self {
        Self(_mm_loadu_si128(src as *const __m128i))
    }

    #[inline(always)]
    unsafe fn store(self, dst: *mut u8) {
        _mm_storeu_si128(dst as *mut __m128i, self.0);
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        Self(unsafe { _mm_xor_si128(self.0, other.0) })
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(unsafe { _mm_and_si128(self.0, other.0) })
    }

    #[inline(always)]
    fn xaesl(x: Self, y: Self) -> Self {
        Self(unsafe { _mm_aesenc_si128(_mm_xor_si128(x.0, y.0), _mm_setzero_si128()) })
    }

    /// `aesenc` XORs the key operand after MixColumns, so the output XOR is
    /// free here.
    #[inline(always)]
    fn aeslx(x: Self, y: Self) -> Self {
        Self(unsafe { _mm_aesenc_si128(x.0, y.0) })
    }

    /// Encrypt and decrypt schedule identically here; keep the native
    /// `aesenc` instead of the default's `xaesl_dec` + XOR.
    #[inline(always)]
    fn aeslx_dec(x: Self, y: Self) -> Self {
        Self::aeslx(x, y)
    }

    #[inline(always)]
    fn aesl_xor(x: Self, y: Self, m: Self) -> Self {
        Self(unsafe { _mm_aesenc_si128(_mm_xor_si128(x.0, y.0), m.0) })
    }
}
