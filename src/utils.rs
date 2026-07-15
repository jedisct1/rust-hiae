//! Input validation and small helpers.

use crate::error::{Error, Result};

/// Maximum length for plaintext and associated data (2^61 - 1 bytes).
pub const MAX_DATA_LEN: u64 = (1u64 << 61) - 1;

/// Constant-time comparison of two 16-byte tags.
#[inline]
pub fn ct_eq(a: &[u8; 16], b: &[u8; 16]) -> bool {
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }

    result == 0
}

/// Validate input lengths for encryption.
pub fn validate_encrypt_params(plaintext_len: usize, aad_len: usize) -> Result<()> {
    if (aad_len as u64) > MAX_DATA_LEN {
        return Err(Error::AssociatedDataTooLong);
    }

    if (plaintext_len as u64) > MAX_DATA_LEN {
        return Err(Error::PlaintextTooLong);
    }

    Ok(())
}

/// Validate input lengths for decryption.
pub fn validate_decrypt_params(ciphertext_len: usize, aad_len: usize) -> Result<()> {
    if (aad_len as u64) > MAX_DATA_LEN {
        return Err(Error::AssociatedDataTooLong);
    }

    if (ciphertext_len as u64) > MAX_DATA_LEN {
        return Err(Error::CiphertextTooLong);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ct_eq() {
        assert!(ct_eq(&[1u8; 16], &[1u8; 16]));

        let mut other = [1u8; 16];
        other[15] ^= 1;
        assert!(!ct_eq(&[1u8; 16], &other));
    }

    #[test]
    fn test_validate_params() {
        assert!(validate_encrypt_params(100, 200).is_ok());
        assert!(validate_decrypt_params(100, 200).is_ok());
    }
}
