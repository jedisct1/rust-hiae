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

/// Zero-pad data to a multiple of block_size_bits.
#[inline]
pub fn zero_pad(data: &[u8], block_size_bits: usize) -> Vec<u8> {
    let block_size_bytes = block_size_bits / 8;
    let len = data.len();

    if len % block_size_bytes == 0 {
        data.to_vec()
    } else {
        let padding_needed = block_size_bytes - (len % block_size_bytes);
        let mut padded = Vec::with_capacity(len + padding_needed);
        padded.extend_from_slice(data);
        padded.resize(len + padding_needed, 0);
        padded
    }
}

/// Truncate data to n bits.
#[inline]
pub fn truncate(data: &[u8], n_bits: usize) -> Vec<u8> {
    let n_bytes = n_bits / 8;

    if n_bits % 8 == 0 {
        data[..n_bytes.min(data.len())].to_vec()
    } else {
        let mut result = data[..(n_bytes + 1).min(data.len())].to_vec();
        if !result.is_empty() {
            let mask = (1u8 << (n_bits % 8)) - 1;
            if let Some(last) = result.last_mut() {
                *last &= mask;
            }
        }
        result
    }
}

/// Split data into blocks of specified size, ignoring partial blocks.
#[inline]
pub fn split_blocks(data: &[u8], block_size_bytes: usize) -> Vec<[u8; 16]> {
    debug_assert_eq!(block_size_bytes, 16);

    let mut blocks = Vec::new();
    let mut chunks = data.chunks_exact(block_size_bytes);

    for chunk in chunks.by_ref() {
        let mut block = [0u8; 16];
        block.copy_from_slice(chunk);
        blocks.push(block);
    }

    blocks
}

/// Get the last n bits of data as bytes.
#[inline]
pub fn tail(data: &[u8], n_bits: usize) -> Vec<u8> {
    let n_bytes = n_bits / 8;
    let n_bits_partial = n_bits % 8;

    if data.len() == 0 {
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

/// XOR two 16-byte blocks.
#[inline]
pub fn xor_block(a: &[u8; 16], b: &[u8; 16]) -> [u8; 16] {
    let mut result = [0u8; 16];
    for i in 0..16 {
        result[i] = a[i] ^ b[i];
    }
    result
}

/// XOR two byte slices of equal length.
#[inline]
pub fn xor_bytes(a: &[u8], b: &[u8]) -> Vec<u8> {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
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
    fn test_zero_pad() {
        let data = &[1, 2, 3];
        let padded = zero_pad(data, 128);
        assert_eq!(padded.len(), 16);
        assert_eq!(&padded[..3], data);
        assert_eq!(&padded[3..], &[0; 13]);

        // Already aligned
        let data = &[1u8; 16];
        let padded = zero_pad(data, 128);
        assert_eq!(padded.len(), 16);
        assert_eq!(padded, data);
    }

    #[test]
    fn test_truncate() {
        let data = &[0xff; 10];

        // Truncate to full bytes
        let result = truncate(data, 64);
        assert_eq!(result.len(), 8);
        assert_eq!(result, vec![0xff; 8]);

        // Truncate to partial byte
        let result = truncate(data, 12);
        assert_eq!(result.len(), 2);
        assert_eq!(result, vec![0xff, 0x0f]);
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
