//! HiAE engine, generic over a 128-bit block type.
//!
//! On the hardware backends, the sixteen-block state stays in vector
//! registers across a whole message.
//! Instead of rotating the state after each block, the main loops process
//! 256 bytes per iteration with the rotation baked into compile-time offsets,
//! mirroring the reference C implementation.

#![allow(unsafe_code)]

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
mod aarch64;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "aes",
    target_feature = "sse2"
))]
mod x86_64;

#[cfg(any(
    test,
    not(any(
        all(
            target_arch = "aarch64",
            target_feature = "neon",
            target_feature = "aes"
        ),
        all(
            target_arch = "x86_64",
            target_feature = "aes",
            target_feature = "sse2"
        )
    ))
))]
mod soft;

#[cfg(all(
    target_arch = "aarch64",
    target_feature = "neon",
    target_feature = "aes"
))]
type Platform = aarch64::Neon;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "aes",
    target_feature = "sse2"
))]
type Platform = x86_64::Aesni;

#[cfg(not(any(
    all(
        target_arch = "aarch64",
        target_feature = "neon",
        target_feature = "aes"
    ),
    all(
        target_arch = "x86_64",
        target_feature = "aes",
        target_feature = "sse2"
    )
)))]
type Platform = soft::Soft;

/// HiAE state specialized for the best block implementation available.
pub type HiaeState = State<Platform>;

/// A 128-bit block with the operations HiAE needs.
///
/// The two AES helpers expose where a XOR can be fused into the hardware
/// instruction: ARM folds it into the round input, x86 into the round output.
pub trait Block: Copy {
    fn zero() -> Self;

    /// # Safety
    /// `src` must be valid for reading 16 bytes.
    unsafe fn load(src: *const u8) -> Self;

    /// # Safety
    /// `dst` must be valid for writing 16 bytes.
    unsafe fn store(self, dst: *mut u8);

    #[inline(always)]
    fn from_bytes(bytes: &[u8; 16]) -> Self {
        // SAFETY: an array reference is valid for its 16 bytes.
        unsafe { Self::load(bytes.as_ptr()) }
    }

    #[inline(always)]
    fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        // SAFETY: out is 16 bytes.
        unsafe { self.store(out.as_mut_ptr()) };
        out
    }

    fn xor(self, other: Self) -> Self;
    fn and(self, other: Self) -> Self;

    /// AESL(x ^ y), where AESL is one keyless AES round:
    /// MixColumns(ShiftRows(SubBytes(x ^ y))).
    fn xaesl(x: Self, y: Self) -> Self;

    /// `xaesl` for the decryption path.
    /// Split out so backends can schedule the encrypt and decrypt loops
    /// differently.
    #[inline(always)]
    fn xaesl_dec(x: Self, y: Self) -> Self {
        Self::xaesl(x, y)
    }

    /// AESL(x) ^ y.
    #[inline(always)]
    fn aeslx(x: Self, y: Self) -> Self {
        Self::xaesl(x, Self::zero()).xor(y)
    }

    /// `aeslx` for the decryption path.
    /// Defaults to `xaesl_dec` so overriding that one method is enough.
    #[inline(always)]
    fn aeslx_dec(x: Self, y: Self) -> Self {
        Self::xaesl_dec(x, Self::zero()).xor(y)
    }

    /// AESL(x ^ y) ^ m.
    #[inline(always)]
    fn aesl_xor(x: Self, y: Self, m: Self) -> Self {
        Self::xaesl(x, y).xor(m)
    }

    /// Hint that `src` will be read soon.
    /// May point past the end of a buffer; prefetch hints never fault.
    #[inline(always)]
    fn prefetch_read(_src: *const u8) {}

    /// Hint that `dst` will be written soon.
    #[inline(always)]
    fn prefetch_write(_dst: *mut u8) {}
}

const C0: [u8; 16] = [
    0x32, 0x43, 0xF6, 0xA8, 0x88, 0x5A, 0x30, 0x8D, 0x31, 0x31, 0x98, 0xA2, 0xE0, 0x37, 0x07, 0x34,
];
const C1: [u8; 16] = [
    0x4A, 0x40, 0x93, 0x82, 0x22, 0x99, 0xF3, 0x1D, 0x00, 0x82, 0xEF, 0xA9, 0x8E, 0xC4, 0xE6, 0xC8,
];

macro_rules! repeat16 {
    ($step:ident) => {
        $step!(0);
        $step!(1);
        $step!(2);
        $step!(3);
        $step!(4);
        $step!(5);
        $step!(6);
        $step!(7);
        $step!(8);
        $step!(9);
        $step!(10);
        $step!(11);
        $step!(12);
        $step!(13);
        $step!(14);
        $step!(15);
    };
}

/// State update without output: absorbs one input block at rotation offset O.
#[inline(always)]
fn update_offset<B: Block, const O: usize>(s: &mut [B; 16], m: B) {
    let t = B::aesl_xor(s[O % 16], s[(O + 1) % 16], m);
    s[O % 16] = B::aeslx(s[(O + 13) % 16], t);
    s[(O + 3) % 16] = s[(O + 3) % 16].xor(m);
    s[(O + 13) % 16] = s[(O + 13) % 16].xor(m);
}

/// State update producing the ciphertext block for plaintext block `m`.
#[inline(always)]
fn enc_offset<B: Block, const O: usize>(s: &mut [B; 16], m: B) -> B {
    let t = B::aesl_xor(s[O % 16], s[(O + 1) % 16], m);
    let c = t.xor(s[(O + 9) % 16]);
    s[O % 16] = B::aeslx(s[(O + 13) % 16], t);
    s[(O + 3) % 16] = s[(O + 3) % 16].xor(m);
    s[(O + 13) % 16] = s[(O + 13) % 16].xor(m);
    c
}

/// State update producing the plaintext block for ciphertext block `c`.
#[inline(always)]
fn dec_offset<B: Block, const O: usize>(s: &mut [B; 16], c: B) -> B {
    let t = B::xaesl_dec(s[O % 16], s[(O + 1) % 16]);
    let m0 = s[(O + 9) % 16].xor(c);
    s[O % 16] = B::aeslx_dec(s[(O + 13) % 16], m0);
    let m = m0.xor(t);
    s[(O + 3) % 16] = s[(O + 3) % 16].xor(m);
    s[(O + 13) % 16] = s[(O + 13) % 16].xor(m);
    m
}

/// Decrypt the final partial block: recover the keystream, mask it to the
/// ciphertext length, and absorb the zero-padded plaintext.
/// The masking and the state update share one AES round.
#[inline(always)]
fn dec_partial_offset<B: Block, const O: usize>(s: &mut [B; 16], c: B, mask: B) -> B {
    let t = B::xaesl_dec(s[O % 16], s[(O + 1) % 16]);
    let m = t.xor(c).xor(s[(O + 9) % 16]).and(mask);
    s[O % 16] = B::aeslx_dec(s[(O + 13) % 16], t.xor(m));
    s[(O + 3) % 16] = s[(O + 3) % 16].xor(m);
    s[(O + 13) % 16] = s[(O + 13) % 16].xor(m);
    m
}

/// One full pass over the state, alternating x0 and x1.
#[inline(always)]
fn full_update<B: Block>(s: &mut [B; 16], x0: B, x1: B) {
    macro_rules! step {
        ($i:literal) => {
            update_offset::<B, $i>(s, if $i % 2 == 0 { x0 } else { x1 });
        };
    }
    repeat16!(step);
}

/// Absorb 256 bytes of associated data.
///
/// # Safety
/// `ad` must be valid for reading 256 bytes.
#[inline(always)]
unsafe fn ad_chunk<B: Block>(s: &mut [B; 16], ad: *const u8) {
    B::prefetch_read(ad.add(128));
    B::prefetch_read(ad.add(256));
    let mut m = [B::zero(); 16];
    macro_rules! load {
        ($i:literal) => {
            m[$i] = B::load(ad.add(16 * $i));
        };
    }
    repeat16!(load);
    macro_rules! step {
        ($i:literal) => {
            update_offset::<B, $i>(s, m[$i]);
        };
    }
    repeat16!(step);
}

/// Encrypt 256 bytes.
///
/// # Safety
/// `mi` must be valid for reading and `ci` for writing 256 bytes.
#[inline(always)]
unsafe fn enc_chunk<B: Block>(s: &mut [B; 16], mi: *const u8, ci: *mut u8) {
    B::prefetch_read(mi.add(128));
    B::prefetch_write(ci.add(128));
    B::prefetch_read(mi.add(256));
    B::prefetch_write(ci.add(256));
    let mut m = [B::zero(); 16];
    macro_rules! load {
        ($i:literal) => {
            m[$i] = B::load(mi.add(16 * $i));
        };
    }
    repeat16!(load);
    let mut c = [B::zero(); 16];
    macro_rules! step {
        ($i:literal) => {
            c[$i] = enc_offset::<B, $i>(s, m[$i]);
        };
    }
    repeat16!(step);
    macro_rules! store {
        ($i:literal) => {
            c[$i].store(ci.add(16 * $i));
        };
    }
    repeat16!(store);
}

/// Decrypt 256 bytes.
///
/// # Safety
/// `ci` must be valid for reading and `mi` for writing 256 bytes.
#[inline(always)]
unsafe fn dec_chunk<B: Block>(s: &mut [B; 16], ci: *const u8, mi: *mut u8) {
    B::prefetch_read(ci.add(128));
    B::prefetch_write(mi.add(128));
    B::prefetch_read(ci.add(256));
    B::prefetch_write(mi.add(256));
    let mut c = [B::zero(); 16];
    macro_rules! load {
        ($i:literal) => {
            c[$i] = B::load(ci.add(16 * $i));
        };
    }
    repeat16!(load);
    let mut m = [B::zero(); 16];
    macro_rules! step {
        ($i:literal) => {
            m[$i] = dec_offset::<B, $i>(s, c[$i]);
        };
    }
    repeat16!(step);
    macro_rules! store {
        ($i:literal) => {
            m[$i].store(mi.add(16 * $i));
        };
    }
    repeat16!(store);
}

/// Dispatch an offset-specialized step function on a runtime offset.
/// Tail blocks are processed at increasing offsets without rotating the
/// state; a single rotation happens once the tail is done.
macro_rules! dispatch_offset {
    ($func:ident, $s:expr, $o:expr $(, $x:expr)+) => {
        match $o {
            0 => $func::<B, 0>($s $(, $x)+),
            1 => $func::<B, 1>($s $(, $x)+),
            2 => $func::<B, 2>($s $(, $x)+),
            3 => $func::<B, 3>($s $(, $x)+),
            4 => $func::<B, 4>($s $(, $x)+),
            5 => $func::<B, 5>($s $(, $x)+),
            6 => $func::<B, 6>($s $(, $x)+),
            7 => $func::<B, 7>($s $(, $x)+),
            8 => $func::<B, 8>($s $(, $x)+),
            9 => $func::<B, 9>($s $(, $x)+),
            10 => $func::<B, 10>($s $(, $x)+),
            11 => $func::<B, 11>($s $(, $x)+),
            12 => $func::<B, 12>($s $(, $x)+),
            13 => $func::<B, 13>($s $(, $x)+),
            14 => $func::<B, 14>($s $(, $x)+),
            _ => $func::<B, 15>($s $(, $x)+),
        }
    };
}

#[inline(always)]
fn shift_by<B: Block>(s: &mut [B; 16], n: usize) {
    s.rotate_left(n % 16);
}

/// HiAE state: sixteen 128-bit blocks.
pub struct State<B: Block> {
    s: [B; 16],
}

impl<B: Block> State<B> {
    pub fn new(key: &[u8; 32], nonce: &[u8; 16]) -> Self {
        let c0 = B::from_bytes(&C0);
        let c1 = B::from_bytes(&C1);
        let k0 = B::from_bytes(key[..16].try_into().unwrap());
        let k1 = B::from_bytes(key[16..].try_into().unwrap());
        let n = B::from_bytes(nonce);
        let ze = B::zero();

        let mut s = [
            c0,
            k0,
            c0,
            n,
            ze,
            k0,
            ze,
            c1,
            k1,
            ze,
            n.xor(k1),
            c0,
            c1,
            k1,
            ze,
            c0.xor(c1),
        ];
        full_update(&mut s, k0, k1);
        full_update(&mut s, k0, k1);
        Self { s }
    }

    pub fn absorb(&mut self, ad: &[u8]) {
        if ad.is_empty() {
            return;
        }
        let mut s = self.s;
        let len = ad.len();
        let mut i = 0;
        while i + 256 <= len {
            unsafe { ad_chunk(&mut s, ad.as_ptr().add(i)) };
            i += 256;
        }
        let mut o = 0;
        while i + 16 <= len {
            let m = B::from_bytes(ad[i..i + 16].try_into().unwrap());
            dispatch_offset!(update_offset, &mut s, o, m);
            o += 1;
            i += 16;
        }
        if i < len {
            let mut buf = [0u8; 16];
            buf[..len - i].copy_from_slice(&ad[i..]);
            dispatch_offset!(update_offset, &mut s, o, B::from_bytes(&buf));
            o += 1;
        }
        shift_by(&mut s, o);
        self.s = s;
    }

    /// Encrypt `mi` into `ci`.
    ///
    /// # Safety
    /// `ci` must be valid for writing `mi.len()` bytes and must not
    /// overlap `mi`.
    pub unsafe fn enc(&mut self, ci: *mut u8, mi: &[u8]) {
        if mi.is_empty() {
            return;
        }
        let mut s = self.s;
        let len = mi.len();
        let mut i = 0;
        while i + 256 <= len {
            enc_chunk(&mut s, mi.as_ptr().add(i), ci.add(i));
            i += 256;
        }
        let mut o = 0;
        while i + 16 <= len {
            let m = B::load(mi.as_ptr().add(i));
            let c = dispatch_offset!(enc_offset, &mut s, o, m);
            c.store(ci.add(i));
            o += 1;
            i += 16;
        }
        if i < len {
            let mut buf = [0u8; 16];
            buf[..len - i].copy_from_slice(&mi[i..]);
            let c = dispatch_offset!(enc_offset, &mut s, o, B::from_bytes(&buf));
            o += 1;
            let cb = c.to_bytes();
            core::ptr::copy_nonoverlapping(cb.as_ptr(), ci.add(i), len - i);
        }
        shift_by(&mut s, o);
        self.s = s;
    }

    /// Decrypt `ci` into `mi`.
    ///
    /// # Safety
    /// `mi` must be valid for writing `ci.len()` bytes and must not
    /// overlap `ci`.
    pub unsafe fn dec(&mut self, mi: *mut u8, ci: &[u8]) {
        if ci.is_empty() {
            return;
        }
        let mut s = self.s;
        let len = ci.len();
        let mut i = 0;
        while i + 256 <= len {
            dec_chunk(&mut s, ci.as_ptr().add(i), mi.add(i));
            i += 256;
        }
        let mut o = 0;
        while i + 16 <= len {
            let c = B::load(ci.as_ptr().add(i));
            let m = dispatch_offset!(dec_offset, &mut s, o, c);
            m.store(mi.add(i));
            o += 1;
            i += 16;
        }
        if i < len {
            let mut buf = [0u8; 16];
            buf[..len - i].copy_from_slice(&ci[i..]);
            let mut mask = [0u8; 16];
            mask[..len - i].fill(0xff);
            let m = dispatch_offset!(
                dec_partial_offset,
                &mut s,
                o,
                B::from_bytes(&buf),
                B::from_bytes(&mask)
            );
            o += 1;
            let mb = m.to_bytes();
            core::ptr::copy_nonoverlapping(mb.as_ptr(), mi.add(i), len - i);
        }
        shift_by(&mut s, o);
        self.s = s;
    }

    pub fn finalize(&self, ad_len: u64, msg_len: u64) -> [u8; 16] {
        let mut s = self.s;
        let mut lens = [0u8; 16];
        lens[..8].copy_from_slice(&(ad_len * 8).to_le_bytes());
        lens[8..].copy_from_slice(&(msg_len * 8).to_le_bytes());
        let t = B::from_bytes(&lens);
        full_update(&mut s, t, t);
        full_update(&mut s, t, t);
        let mut acc = s[0];
        for b in &s[1..] {
            acc = acc.xor(*b);
        }
        acc.to_bytes()
    }
}

impl<B: Block> Drop for State<B> {
    fn drop(&mut self) {
        for b in &mut self.s {
            unsafe { core::ptr::write_volatile(b, B::zero()) };
        }
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_soft_vs_platform(msg_len: usize, ad_len: usize) {
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 16];
        let msg: alloc::vec::Vec<u8> = (0..msg_len).map(|i| i as u8).collect();
        let ad: alloc::vec::Vec<u8> = (0..ad_len).map(|i| (i * 7) as u8).collect();

        let mut ct_a = alloc::vec![0u8; msg_len];
        let mut st_a = State::<Platform>::new(&key, &nonce);
        st_a.absorb(&ad);
        unsafe { st_a.enc(ct_a.as_mut_ptr(), &msg) };
        let tag_a = st_a.finalize(ad_len as u64, msg_len as u64);

        let mut ct_b = alloc::vec![0u8; msg_len];
        let mut st_b = State::<soft::Soft>::new(&key, &nonce);
        st_b.absorb(&ad);
        unsafe { st_b.enc(ct_b.as_mut_ptr(), &msg) };
        let tag_b = st_b.finalize(ad_len as u64, msg_len as u64);

        assert_eq!(ct_a, ct_b, "msg_len={msg_len} ad_len={ad_len}");
        assert_eq!(tag_a, tag_b, "msg_len={msg_len} ad_len={ad_len}");

        let mut pt = alloc::vec![0u8; msg_len];
        let mut st_c = State::<Platform>::new(&key, &nonce);
        st_c.absorb(&ad);
        unsafe { st_c.dec(pt.as_mut_ptr(), &ct_a) };
        let tag_c = st_c.finalize(ad_len as u64, msg_len as u64);
        assert_eq!(pt, msg, "msg_len={msg_len} ad_len={ad_len}");
        assert_eq!(tag_c, tag_a, "msg_len={msg_len} ad_len={ad_len}");
    }

    #[test]
    fn soft_and_platform_agree() {
        for msg_len in [0, 1, 15, 16, 17, 255, 256, 257, 511, 512, 1000] {
            for ad_len in [0, 1, 16, 40, 256, 300] {
                roundtrip_soft_vs_platform(msg_len, ad_len);
            }
        }
    }
}
