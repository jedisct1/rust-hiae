//! Core HiAE algorithm implementation.

use crate::error::{Error, Result};
use crate::intrinsics;
use crate::utils::{self, ct_eq, le64, split_blocks, xor_block, zero_pad};
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
    rotation_offset: usize,
}

impl HiaeState {
    /// Create a new zero-initialized state.
    fn new() -> Self {
        Self {
            blocks: [[0u8; 16]; 16],
            rotation_offset: 0,
        }
    }

    /// Get the physical block index from a logical index.
    #[inline]
    fn get_block_index(&self, logical_index: usize) -> usize {
        (logical_index + self.rotation_offset) % 16
    }

    /// Rotate state blocks left by one position (optimized with cycling index).
    #[inline]
    fn rol(&mut self) {
        self.rotation_offset = (self.rotation_offset + 1) % 16;
    }

    /// Core update function.
    #[inline]
    fn update(&mut self, xi: &[u8; 16]) {
        let temp = xor_block(
            &intrinsics::aesl(&xor_block(&self.blocks[self.get_block_index(0)], &self.blocks[self.get_block_index(1)])),
            xi,
        );
        self.blocks[self.get_block_index(0)] = xor_block(&intrinsics::aesl(&self.blocks[self.get_block_index(13)]), &temp);
        self.blocks[self.get_block_index(3)] = xor_block(&self.blocks[self.get_block_index(3)], xi);
        self.blocks[self.get_block_index(13)] = xor_block(&self.blocks[self.get_block_index(13)], xi);
        self.rol();
    }

    /// Update function with encryption.
    #[inline]
    fn update_enc(&mut self, mi: &[u8; 16]) -> [u8; 16] {
        let temp = xor_block(
            &intrinsics::aesl(&xor_block(&self.blocks[self.get_block_index(0)], &self.blocks[self.get_block_index(1)])),
            mi,
        );
        let ci = xor_block(&temp, &self.blocks[self.get_block_index(9)]);
        self.blocks[self.get_block_index(0)] = xor_block(&intrinsics::aesl(&self.blocks[self.get_block_index(13)]), &temp);
        self.blocks[self.get_block_index(3)] = xor_block(&self.blocks[self.get_block_index(3)], mi);
        self.blocks[self.get_block_index(13)] = xor_block(&self.blocks[self.get_block_index(13)], mi);
        self.rol();
        ci
    }

    /// Update function with decryption.
    #[inline]
    fn update_dec(&mut self, ci: &[u8; 16]) -> [u8; 16] {
        let temp = xor_block(ci, &self.blocks[self.get_block_index(9)]);
        let mi = xor_block(
            &intrinsics::aesl(&xor_block(&self.blocks[self.get_block_index(0)], &self.blocks[self.get_block_index(1)])),
            &temp,
        );
        self.blocks[self.get_block_index(0)] = xor_block(&intrinsics::aesl(&self.blocks[self.get_block_index(13)]), &temp);
        self.blocks[self.get_block_index(3)] = xor_block(&self.blocks[self.get_block_index(3)], &mi);
        self.blocks[self.get_block_index(13)] = xor_block(&self.blocks[self.get_block_index(13)], &mi);
        self.rol();
        mi
    }

    /// Apply 32 update rounds for full diffusion.
    #[inline]
    fn diffuse(&mut self, x: &[u8; 16]) {
        for _ in 0..32 {
            self.update(x);
        }
    }

    /// Initialize state from key and nonce.
    fn init(&mut self, key: &[u8; 32], nonce: &[u8; 16]) {
        // Reset rotation offset
        self.rotation_offset = 0;

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
        self.blocks[self.get_block_index(9)] = xor_block(&self.blocks[self.get_block_index(9)], &k0);
        self.blocks[self.get_block_index(13)] = xor_block(&self.blocks[self.get_block_index(13)], &k1);
    }

    /// Absorb a block of associated data.
    #[inline]
    fn absorb(&mut self, ai: &[u8; 16]) {
        self.update(ai);
    }

    /// Encrypt a single block.
    #[inline]
    fn enc(&mut self, mi: &[u8; 16]) -> [u8; 16] {
        self.update_enc(mi)
    }

    /// Decrypt a single block.
    #[inline]
    fn dec(&mut self, ci: &[u8; 16]) -> [u8; 16] {
        self.update_dec(ci)
    }

    /// Decrypt a partial block.
    fn dec_partial(&mut self, cn: &[u8]) -> Vec<u8> {
        // Step 1: Recover keystream
        let mut zero_block = [0u8; 16];
        zero_block[..cn.len()].copy_from_slice(cn);

        let ks = xor_block(
            &xor_block(
                &intrinsics::aesl(&xor_block(&self.blocks[self.get_block_index(0)], &self.blocks[self.get_block_index(1)])),
                &zero_block,
            ),
            &self.blocks[self.get_block_index(9)],
        );

        // Step 2: Construct full ciphertext block
        let tail_bits = 128 - (cn.len() * 8);
        let tail_bytes = utils::tail(&ks, tail_bits);

        let mut ci_block = [0u8; 16];
        ci_block[..cn.len()].copy_from_slice(cn);
        ci_block[cn.len()..].copy_from_slice(&tail_bytes[..16 - cn.len()]);

        // Step 3: Decrypt full block
        let mi = self.update_dec(&ci_block);

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

    // Process associated data
    if !aad.is_empty() {
        let ad_padded = zero_pad(aad, 128);
        let ad_blocks = split_blocks(&ad_padded, 16);
        for block in ad_blocks {
            state.absorb(&block);
        }
    }

    // Encrypt plaintext
    if !plaintext.is_empty() {
        let msg_padded = zero_pad(plaintext, 128);
        let msg_blocks = split_blocks(&msg_padded, 16);
        for block in msg_blocks {
            let encrypted_block = state.enc(&block);
            ciphertext.extend_from_slice(&encrypted_block);
        }

        // Truncate ciphertext to original plaintext length
        ciphertext.truncate(plaintext.len());
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

    // Process associated data
    if !aad.is_empty() {
        let ad_padded = zero_pad(aad, 128);
        let ad_blocks = split_blocks(&ad_padded, 16);
        for block in ad_blocks {
            state.absorb(&block);
        }
    }

    // Decrypt ciphertext
    if !ciphertext.is_empty() {
        let ct_blocks = split_blocks(ciphertext, 16);
        let remainder = ciphertext.len() % 16;

        for block in ct_blocks {
            let decrypted_block = state.dec(&block);
            plaintext.extend_from_slice(&decrypted_block);
        }

        // Handle partial block if present
        if remainder != 0 {
            let partial_start = ciphertext.len() - remainder;
            let partial_ct = &ciphertext[partial_start..];
            let partial_pt = state.dec_partial(partial_ct);
            plaintext.extend_from_slice(&partial_pt);
        }

        // Truncate plaintext to original ciphertext length
        plaintext.truncate(ciphertext.len());
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
