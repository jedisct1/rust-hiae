//! Core HiAE algorithm implementation.

use crate::error::{Error, Result};
use crate::intrinsics;
use crate::utils::{self, ct_eq, le64, xor_block};
use alloc::vec::Vec;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// HiAE constants C0 and C1 (domain separation constants).
const C0: [u8; 16] = [
    0x32, 0x43, 0xF6, 0xA8, 0x88, 0x5A, 0x30, 0x8D, 0x31, 0x31, 0x98, 0xA2, 0xE0, 0x37, 0x07, 0x34,
];
const C1: [u8; 16] = [
    0x4A, 0x40, 0x93, 0x82, 0x22, 0x99, 0xF3, 0x1D, 0x00, 0x82, 0xEF, 0xA9, 0x8E, 0xC4, 0xE6, 0xC8,
];

/// HiAE state containing sixteen 128-bit blocks.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
struct HiaeState {
    blocks: [[u8; 16]; 16],
}

impl HiaeState {
    /// Create a new zero-initialized state.
    fn new() -> Self {
        Self {
            blocks: [[0u8; 16]; 16],
        }
    }

    /// Rotate state blocks left by one position.
    #[inline]
    fn rol(&mut self) {
        let temp = self.blocks[0];
        for i in 0..15 {
            self.blocks[i] = self.blocks[i + 1];
        }
        self.blocks[15] = temp;
    }

    /// Core update function with compile-time indexing.
    #[inline]
    fn update<const I: usize>(&mut self, xi: &[u8; 16]) {
        let temp = xor_block(
            &intrinsics::aesl(&xor_block(&self.blocks[I], &self.blocks[(I + 1) % 16])),
            xi,
        );
        self.blocks[I] = xor_block(&intrinsics::aesl(&self.blocks[(I + 13) % 16]), &temp);
        self.blocks[(I + 3) % 16] = xor_block(&self.blocks[(I + 3) % 16], xi);
        self.blocks[(I + 13) % 16] = xor_block(&self.blocks[(I + 13) % 16], xi);
    }

    /// Update function with encryption and compile-time indexing.
    #[inline]
    fn update_enc<const I: usize>(&mut self, mi: &[u8; 16]) -> [u8; 16] {
        let temp = xor_block(
            &intrinsics::aesl(&xor_block(&self.blocks[I], &self.blocks[(I + 1) % 16])),
            mi,
        );
        let ci = xor_block(&temp, &self.blocks[(I + 9) % 16]);
        self.blocks[I] = xor_block(&intrinsics::aesl(&self.blocks[(I + 13) % 16]), &temp);
        self.blocks[(I + 3) % 16] = xor_block(&self.blocks[(I + 3) % 16], mi);
        self.blocks[(I + 13) % 16] = xor_block(&self.blocks[(I + 13) % 16], mi);
        ci
    }

    /// Update function with decryption and compile-time indexing.
    #[inline]
    fn update_dec<const I: usize>(&mut self, ci: &[u8; 16]) -> [u8; 16] {
        let temp = xor_block(ci, &self.blocks[(I + 9) % 16]);
        let mi = xor_block(
            &intrinsics::aesl(&xor_block(&self.blocks[I], &self.blocks[(I + 1) % 16])),
            &temp,
        );
        self.blocks[I] = xor_block(&intrinsics::aesl(&self.blocks[(I + 13) % 16]), &temp);
        self.blocks[(I + 3) % 16] = xor_block(&self.blocks[(I + 3) % 16], &mi);
        self.blocks[(I + 13) % 16] = xor_block(&self.blocks[(I + 13) % 16], &mi);
        mi
    }

    /// Apply 32 update rounds for full diffusion.
    #[inline]
    fn diffuse(&mut self, x: &[u8; 16]) {
        for _ in 0..2 {
            self.update::<0>(x);
            self.update::<1>(x);
            self.update::<2>(x);
            self.update::<3>(x);
            self.update::<4>(x);
            self.update::<5>(x);
            self.update::<6>(x);
            self.update::<7>(x);
            self.update::<8>(x);
            self.update::<9>(x);
            self.update::<10>(x);
            self.update::<11>(x);
            self.update::<12>(x);
            self.update::<13>(x);
            self.update::<14>(x);
            self.update::<15>(x);
        }
    }

    /// Initialize state from key and nonce.
    fn init(&mut self, key: &[u8; 32], nonce: &[u8; 16]) {
        // Split key into two 128-bit halves
        let mut k0 = [0u8; 16];
        let mut k1 = [0u8; 16];
        k0.copy_from_slice(&key[..16]);
        k1.copy_from_slice(&key[16..]);

        // Initialize state blocks according to specification
        self.blocks[0] = C0;
        self.blocks[1] = k1;
        self.blocks[2] = *nonce;
        self.blocks[3] = C0;
        self.blocks[4] = [0u8; 16];
        self.blocks[5] = xor_block(nonce, &k0);
        self.blocks[6] = [0u8; 16];
        self.blocks[7] = C1;
        self.blocks[8] = xor_block(nonce, &k1);
        self.blocks[9] = [0u8; 16];
        self.blocks[10] = k1;
        self.blocks[11] = C0;
        self.blocks[12] = C1;
        self.blocks[13] = k1;
        self.blocks[14] = [0u8; 16];
        self.blocks[15] = xor_block(&C0, &C1);

        // Diffuse with C0
        self.diffuse(&C0);

        // Final XORs
        self.blocks[9] = xor_block(&self.blocks[9], &k0);
        self.blocks[13] = xor_block(&self.blocks[13], &k1);
    }

    /// Absorb a batch of 16 blocks of associated data.
    #[inline]
    fn absorb_batch(&mut self, ai: &[[u8; 16]; 16]) {
        self.update::<0>(&ai[0]);
        self.update::<1>(&ai[1]);
        self.update::<2>(&ai[2]);
        self.update::<3>(&ai[3]);
        self.update::<4>(&ai[4]);
        self.update::<5>(&ai[5]);
        self.update::<6>(&ai[6]);
        self.update::<7>(&ai[7]);
        self.update::<8>(&ai[8]);
        self.update::<9>(&ai[9]);
        self.update::<10>(&ai[10]);
        self.update::<11>(&ai[11]);
        self.update::<12>(&ai[12]);
        self.update::<13>(&ai[13]);
        self.update::<14>(&ai[14]);
        self.update::<15>(&ai[15]);
    }

    /// Absorb a single block of associated data.
    #[inline]
    fn absorb(&mut self, ai: &[u8; 16]) {
        self.update::<0>(ai);
        self.rol();
    }

    /// Encrypt a batch of 16 blocks.
    #[inline]
    fn enc_batch(&mut self, mi: &[[u8; 16]; 16]) -> [[u8; 16]; 16] {
        [
            self.update_enc::<0>(&mi[0]),
            self.update_enc::<1>(&mi[1]),
            self.update_enc::<2>(&mi[2]),
            self.update_enc::<3>(&mi[3]),
            self.update_enc::<4>(&mi[4]),
            self.update_enc::<5>(&mi[5]),
            self.update_enc::<6>(&mi[6]),
            self.update_enc::<7>(&mi[7]),
            self.update_enc::<8>(&mi[8]),
            self.update_enc::<9>(&mi[9]),
            self.update_enc::<10>(&mi[10]),
            self.update_enc::<11>(&mi[11]),
            self.update_enc::<12>(&mi[12]),
            self.update_enc::<13>(&mi[13]),
            self.update_enc::<14>(&mi[14]),
            self.update_enc::<15>(&mi[15]),
        ]
    }

    /// Encrypt a single block.
    #[inline]
    fn enc(&mut self, mi: &[u8; 16]) -> [u8; 16] {
        let result = self.update_enc::<0>(mi);
        self.rol();
        result
    }

    /// Decrypt a batch of 16 blocks.
    #[inline]
    fn dec_batch(&mut self, ci: &[[u8; 16]; 16]) -> [[u8; 16]; 16] {
        [
            self.update_dec::<0>(&ci[0]),
            self.update_dec::<1>(&ci[1]),
            self.update_dec::<2>(&ci[2]),
            self.update_dec::<3>(&ci[3]),
            self.update_dec::<4>(&ci[4]),
            self.update_dec::<5>(&ci[5]),
            self.update_dec::<6>(&ci[6]),
            self.update_dec::<7>(&ci[7]),
            self.update_dec::<8>(&ci[8]),
            self.update_dec::<9>(&ci[9]),
            self.update_dec::<10>(&ci[10]),
            self.update_dec::<11>(&ci[11]),
            self.update_dec::<12>(&ci[12]),
            self.update_dec::<13>(&ci[13]),
            self.update_dec::<14>(&ci[14]),
            self.update_dec::<15>(&ci[15]),
        ]
    }

    /// Decrypt a single block.
    #[inline]
    fn dec(&mut self, ci: &[u8; 16]) -> [u8; 16] {
        let result = self.update_dec::<0>(ci);
        self.rol();
        result
    }

    /// Decrypt a partial block.
    fn dec_partial(&mut self, cn: &[u8]) -> Vec<u8> {
        // Step 1: Recover keystream
        let mut zero_block = [0u8; 16];
        zero_block[..cn.len()].copy_from_slice(cn);

        let ks = xor_block(
            &xor_block(
                &intrinsics::aesl(&xor_block(&self.blocks[0], &self.blocks[1])),
                &zero_block,
            ),
            &self.blocks[9],
        );

        // Step 2: Construct full ciphertext block
        let tail_bits = 128 - (cn.len() * 8);
        let tail_bytes = utils::tail(&ks, tail_bits);

        let mut ci_block = [0u8; 16];
        ci_block[..cn.len()].copy_from_slice(cn);
        ci_block[cn.len()..].copy_from_slice(&tail_bytes[..16 - cn.len()]);

        // Step 3: Decrypt full block
        let mi = self.update_dec::<0>(&ci_block);
        self.rol();

        // Step 4: Extract partial plaintext
        let mut result = Vec::with_capacity(cn.len());
        result.extend_from_slice(&mi[..cn.len()]);
        result
    }

    /// Generate authentication tag.
    fn finalize(&mut self, ad_len_bits: u64, msg_len_bits: u64) -> [u8; 16] {
        // Create length encoding block
        let ad_len_bytes = le64(ad_len_bits);
        let msg_len_bytes = le64(msg_len_bits);

        let mut t = [0u8; 16];
        t[..8].copy_from_slice(&ad_len_bytes);
        t[8..].copy_from_slice(&msg_len_bytes);

        self.diffuse(&t);

        // XOR all state blocks using vectorized reduction
        intrinsics::xor_reduce_blocks(&self.blocks)
    }
}

/// Encrypt plaintext with associated data using HiAE.
pub fn encrypt(
    plaintext: &[u8],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
) -> Result<(Vec<u8>, [u8; 16])> {
    // Validate input parameters
    utils::validate_params(plaintext.len(), aad.len(), key, nonce)?;

    let mut state = HiaeState::new();
    state.init(key, nonce);

    // Pre-allocate ciphertext with exact capacity
    let mut ciphertext = Vec::with_capacity(plaintext.len());

    // Process associated data with batch processing
    if !aad.is_empty() {
        let mut i = 0;
        // Process full 16-block batches (256 bytes)
        while i + 256 <= aad.len() {
            let mut batch = [[0u8; 16]; 16];
            for j in 0..16 {
                batch[j].copy_from_slice(&aad[i + j * 16..i + (j + 1) * 16]);
            }
            state.absorb_batch(&batch);
            i += 256;
        }
        // Process remaining full blocks
        while i + 16 <= aad.len() {
            let mut block = [0u8; 16];
            block.copy_from_slice(&aad[i..i + 16]);
            state.absorb(&block);
            i += 16;
        }
        // Process partial block
        if i < aad.len() {
            let mut block = [0u8; 16];
            block[..aad.len() - i].copy_from_slice(&aad[i..]);
            state.absorb(&block);
        }
    }

    // Encrypt plaintext with batch processing
    if !plaintext.is_empty() {
        let mut i = 0;
        // Process full 16-block batches (256 bytes)
        while i + 256 <= plaintext.len() {
            let mut batch = [[0u8; 16]; 16];
            for j in 0..16 {
                batch[j].copy_from_slice(&plaintext[i + j * 16..i + (j + 1) * 16]);
            }
            let encrypted_batch = state.enc_batch(&batch);
            for block in encrypted_batch {
                ciphertext.extend_from_slice(&block);
            }
            i += 256;
        }
        // Process remaining full blocks
        while i + 16 <= plaintext.len() {
            let mut block = [0u8; 16];
            block.copy_from_slice(&plaintext[i..i + 16]);
            let encrypted_block = state.enc(&block);
            ciphertext.extend_from_slice(&encrypted_block);
            i += 16;
        }
        // Process partial block
        if i < plaintext.len() {
            let mut block = [0u8; 16];
            block[..plaintext.len() - i].copy_from_slice(&plaintext[i..]);
            let encrypted_block = state.enc(&block);
            ciphertext.extend_from_slice(&encrypted_block[..plaintext.len() - i]);
        }
    }

    // Generate authentication tag
    let tag = state.finalize((aad.len() * 8) as u64, (plaintext.len() * 8) as u64);

    Ok((ciphertext, tag))
}

/// Decrypt ciphertext and verify authentication tag.
pub fn decrypt(
    ciphertext: &[u8],
    tag: &[u8; 16],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
) -> Result<Vec<u8>> {
    // Validate input parameters
    utils::validate_params(ciphertext.len(), aad.len(), key, nonce)?;

    let mut state = HiaeState::new();
    state.init(key, nonce);

    // Pre-allocate plaintext with exact capacity
    let mut plaintext = Vec::with_capacity(ciphertext.len());

    // Process associated data with batch processing
    if !aad.is_empty() {
        let mut i = 0;
        // Process full 16-block batches (256 bytes)
        while i + 256 <= aad.len() {
            let mut batch = [[0u8; 16]; 16];
            for j in 0..16 {
                batch[j].copy_from_slice(&aad[i + j * 16..i + (j + 1) * 16]);
            }
            state.absorb_batch(&batch);
            i += 256;
        }
        // Process remaining full blocks
        while i + 16 <= aad.len() {
            let mut block = [0u8; 16];
            block.copy_from_slice(&aad[i..i + 16]);
            state.absorb(&block);
            i += 16;
        }
        // Process partial block
        if i < aad.len() {
            let mut block = [0u8; 16];
            block[..aad.len() - i].copy_from_slice(&aad[i..]);
            state.absorb(&block);
        }
    }

    // Decrypt ciphertext with batch processing
    if !ciphertext.is_empty() {
        let mut i = 0;
        // Process full 16-block batches (256 bytes)
        while i + 256 <= ciphertext.len() {
            let mut batch = [[0u8; 16]; 16];
            for j in 0..16 {
                batch[j].copy_from_slice(&ciphertext[i + j * 16..i + (j + 1) * 16]);
            }
            let decrypted_batch = state.dec_batch(&batch);
            for block in decrypted_batch {
                plaintext.extend_from_slice(&block);
            }
            i += 256;
        }
        // Process remaining full blocks
        while i + 16 <= ciphertext.len() {
            let mut block = [0u8; 16];
            block.copy_from_slice(&ciphertext[i..i + 16]);
            let decrypted_block = state.dec(&block);
            plaintext.extend_from_slice(&decrypted_block);
            i += 16;
        }
        // Process partial block
        if i < ciphertext.len() {
            let partial_pt = state.dec_partial(&ciphertext[i..]);
            plaintext.extend_from_slice(&partial_pt);
        }
    }

    // Generate expected tag
    let expected_tag = state.finalize((aad.len() * 8) as u64, (ciphertext.len() * 8) as u64);

    // Verify tag in constant time
    if !ct_eq(tag, &expected_tag) {
        return Err(Error::AuthenticationFailed);
    }

    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_operations() {
        let mut state = HiaeState::new();
        let key = [0u8; 32];
        let nonce = [0u8; 16];

        state.init(&key, &nonce);

        // Test basic operations don't panic
        let block = [0x55u8; 16];
        state.absorb(&block);

        let encrypted = state.enc(&block);
        assert_ne!(encrypted, block); // Should be different

        // Reset state and decrypt
        state.init(&key, &nonce);
        state.absorb(&block);
        let decrypted = state.dec(&encrypted);
        assert_eq!(decrypted, block);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = [0x01u8; 32];
        let nonce = [0x02u8; 16];
        let plaintext = b"Hello, HiAE!";
        let aad = b"associated data";

        let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce).unwrap();
        let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce).unwrap();

        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_empty_inputs() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];
        let plaintext = b"";
        let aad = b"";

        let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce).unwrap();
        assert!(ciphertext.is_empty());

        let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_authentication_failure() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];
        let plaintext = b"secret message";
        let aad = b"public header";

        let (ciphertext, mut tag) = encrypt(plaintext, aad, &key, &nonce).unwrap();

        // Corrupt the tag
        tag[0] ^= 1;

        let result = decrypt(&ciphertext, &tag, aad, &key, &nonce);
        assert!(matches!(result, Err(Error::AuthenticationFailed)));
    }
}
