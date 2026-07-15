//! NEON block for aarch64 with the crypto extensions.
//!
//! This module is only compiled when `aes` and `neon` are enabled at compile
//! time, so the intrinsics can be used without runtime detection and inline
//! freely into the generic engine.

use super::Block;
use core::arch::aarch64::*;

#[derive(Copy, Clone)]
pub struct Neon(uint8x16_t);

impl Block for Neon {
    #[inline(always)]
    fn zero() -> Self {
        Self(unsafe { vmovq_n_u8(0) })
    }

    #[inline(always)]
    unsafe fn load(src: *const u8) -> Self {
        Self(vld1q_u8(src))
    }

    #[inline(always)]
    unsafe fn store(self, dst: *mut u8) {
        vst1q_u8(dst, self.0);
    }

    #[inline(always)]
    fn xor(self, other: Self) -> Self {
        Self(unsafe { veorq_u8(self.0, other.0) })
    }

    #[inline(always)]
    fn and(self, other: Self) -> Self {
        Self(unsafe { vandq_u8(self.0, other.0) })
    }

    /// `vaeseq_u8` XORs the key operand before SubBytes/ShiftRows, so the
    /// input XOR is free here.
    #[inline(always)]
    fn xaesl(x: Self, y: Self) -> Self {
        Self(unsafe { vaesmcq_u8(vaeseq_u8(x.0, y.0)) })
    }

    /// Inline asm keeps the aese/aesmc pair adjacent: Apple cores macro-fuse
    /// the pair and LLVM's scheduler tends to split it in the decrypt loop.
    /// The encrypt loop schedules better from plain intrinsics, so only the
    /// decrypt path uses this variant.
    #[inline(always)]
    fn xaesl_dec(x: Self, y: Self) -> Self {
        let mut out = x.0;
        unsafe {
            core::arch::asm!(
                "aese {x:v}.16b, {y:v}.16b",
                "aesmc {x:v}.16b, {x:v}.16b",
                x = inout(vreg) out,
                y = in(vreg) y.0,
                options(pure, nomem, nostack, preserves_flags),
            );
        }
        Self(out)
    }

    #[inline(always)]
    fn prefetch_read(src: *const u8) {
        unsafe {
            core::arch::asm!("prfm pldl1strm, [{0}]", in(reg) src, options(nostack, preserves_flags));
        }
    }

    #[inline(always)]
    fn prefetch_write(dst: *mut u8) {
        unsafe {
            core::arch::asm!("prfm pstl1strm, [{0}]", in(reg) dst, options(nostack, preserves_flags));
        }
    }
}
