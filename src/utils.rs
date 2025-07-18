//! Utility functions for byte manipulation and data processing.

use crate::error::{Error, Result};
use alloc::vec::Vec;

/// Maximum length for plaintext and associated data (2^61 - 1 bytes).
pub const MAX_DATA_LEN: u64 = (1u64 << 61) - 1;

/// Convert a 64-bit integer to little-endian bytes.
#[inline]
pub fn le64(n: u64) -> [u8; 8] {
    n.to_le_bytes()
}

/// Get the last n bits of data as bytes.
#[inline]
pub fn tail(data: &[u8], n_bits: usize) -> Vec<u8> {
    let n_bytes = n_bits / 8;
    let n_bits_partial = n_bits % 8;

    if data.is_empty() {
        return Vec::new();
    }

    if n_bits_partial == 0 {
        let start = data.len().saturating_sub(n_bytes);
        data[start..].to_vec()
    } else {
        let total_bytes_needed = n_bytes + 1;
        if total_bytes_needed > data.len() {
            data.to_vec()
        } else {
            let start = data.len() - total_bytes_needed;
            let mut result = data[start..].to_vec();

            // Mask the first byte to get only the needed bits
            let shift = 8 - n_bits_partial;
            result[0] = (result[0] >> shift) << shift;
            result
        }
    }
}

/// XOR two 16-byte blocks using SIMD when available.
#[inline]
pub fn xor_block(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    crate::intrinsics::xor_block_simd(a, b)
}

/// Constant-time comparison of two byte arrays.
#[inline]
pub fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }

    result == 0
}

/// Validate input parameters for encryption/decryption.
pub fn validate_params(
    plaintext_len: usize,
    aad_len: usize,
    key: &[u8],
    nonce: &[u8],
) -> Result<()> {
    if key.len() != 32 {
        return Err(Error::InvalidKeyLength);
    }

    if nonce.len() != 16 {
        return Err(Error::InvalidNonceLength);
    }

    if (plaintext_len as u64) > MAX_DATA_LEN {
        return Err(Error::PlaintextTooLong);
    }

    if (aad_len as u64) > MAX_DATA_LEN {
        return Err(Error::AssociatedDataTooLong);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_le64() {
        assert_eq!(
            le64(0x1234567890abcdef),
            [0xef, 0xcd, 0xab, 0x90, 0x78, 0x56, 0x34, 0x12]
        );
        assert_eq!(le64(0), [0; 8]);
    }

    #[test]
    fn test_xor_block() {
        let a = [0xf0; 16];
        let b = [0x0f; 16];
        let result = xor_block(&a, &b);
        assert_eq!(result, [0xff; 16]);
    }

    #[test]
    fn test_ct_eq() {
        assert!(ct_eq(&[1, 2, 3], &[1, 2, 3]));
        assert!(!ct_eq(&[1, 2, 3], &[1, 2, 4]));
        assert!(!ct_eq(&[1, 2], &[1, 2, 3]));
    }

    #[test]
    fn test_validate_params() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];

        assert!(validate_params(100, 200, &key, &nonce).is_ok());

        // Wrong key length
        assert!(validate_params(100, 200, &[0u8; 31], &nonce).is_err());

        // Wrong nonce length
        assert!(validate_params(100, 200, &key, &[0u8; 15]).is_err());
    }
}
