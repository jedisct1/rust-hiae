//! Core HiAE algorithm implementation.

use crate::backend::HiaeState;
use crate::error::{Error, Result};
use crate::utils::{self, ct_eq};
use alloc::vec::Vec;

/// Run the full encryption pipeline, writing the ciphertext to `out`.
///
/// # Safety
/// `out` must be valid for writing `plaintext.len()` bytes and must not
/// overlap the input slices.
#[allow(unsafe_code)]
unsafe fn encrypt_raw(
    plaintext: &[u8],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
    out: *mut u8,
) -> [u8; 16] {
    let mut state = HiaeState::new(key, nonce);
    state.absorb(aad);
    state.enc(out, plaintext);
    state.finalize(aad.len() as u64, plaintext.len() as u64)
}

/// Run the full decryption pipeline, writing the plaintext to `out`,
/// and return the expected tag. The caller checks it.
///
/// # Safety
/// `out` must be valid for writing `ciphertext.len()` bytes and must not
/// overlap the input slices.
#[allow(unsafe_code)]
unsafe fn decrypt_raw(
    ciphertext: &[u8],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
    out: *mut u8,
) -> [u8; 16] {
    let mut state = HiaeState::new(key, nonce);
    state.absorb(aad);
    state.dec(out, ciphertext);
    state.finalize(aad.len() as u64, ciphertext.len() as u64)
}

/// Encrypts plaintext with associated data using HiAE.
///
/// Returns the ciphertext and the 128-bit authentication tag.
/// The key must be uniformly random, and the nonce must never be reused
/// with the same key.
///
/// # Example
///
/// ```rust
/// use hiae::encrypt;
///
/// let key = [0u8; 32];
/// let nonce = [0u8; 16];
/// let plaintext = b"secret message";
/// let aad = b"public header";
///
/// let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce)?;
/// # Ok::<(), hiae::Error>(())
/// ```
#[allow(unsafe_code)]
pub fn encrypt(
    plaintext: &[u8],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
) -> Result<(Vec<u8>, [u8; 16])> {
    utils::validate_encrypt_params(plaintext.len(), aad.len())?;

    let mut ciphertext = Vec::with_capacity(plaintext.len());
    // SAFETY: the reserved capacity covers plaintext.len() bytes and
    // encrypt_raw() initializes all of them before the length is set.
    let tag = unsafe {
        let tag = encrypt_raw(plaintext, aad, key, nonce, ciphertext.as_mut_ptr());
        ciphertext.set_len(plaintext.len());
        tag
    };

    Ok((ciphertext, tag))
}

/// Encrypts plaintext into a caller-provided buffer, avoiding allocation.
///
/// `ciphertext` must be exactly `plaintext.len()` bytes long.
/// Returns the authentication tag on success.
///
/// # Example
///
/// ```rust
/// use hiae::encrypt_into;
///
/// let key = [0u8; 32];
/// let nonce = [0u8; 16];
/// let plaintext = b"secret message";
/// let mut ciphertext = [0u8; 14];
///
/// let tag = encrypt_into(plaintext, b"", &key, &nonce, &mut ciphertext)?;
/// # Ok::<(), hiae::Error>(())
/// ```
#[allow(unsafe_code)]
pub fn encrypt_into(
    plaintext: &[u8],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
    ciphertext: &mut [u8],
) -> Result<[u8; 16]> {
    utils::validate_encrypt_params(plaintext.len(), aad.len())?;
    if ciphertext.len() != plaintext.len() {
        return Err(Error::OutputBufferMismatch);
    }

    // SAFETY: ciphertext is exactly plaintext.len() bytes.
    Ok(unsafe { encrypt_raw(plaintext, aad, key, nonce, ciphertext.as_mut_ptr()) })
}

/// Decrypts ciphertext and verifies the authentication tag.
///
/// Returns the plaintext, or an error if the tag does not match; no
/// plaintext is returned in that case.
/// Tag comparison runs in constant time.
///
/// # Example
///
/// ```rust
/// use hiae::{encrypt, decrypt};
///
/// let key = [0u8; 32];
/// let nonce = [0u8; 16];
/// let plaintext = b"secret message";
/// let aad = b"public header";
///
/// let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce)?;
/// let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce)?;
///
/// assert_eq!(decrypted, plaintext);
/// # Ok::<(), hiae::Error>(())
/// ```
#[allow(unsafe_code)]
pub fn decrypt(
    ciphertext: &[u8],
    tag: &[u8; 16],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
) -> Result<Vec<u8>> {
    utils::validate_decrypt_params(ciphertext.len(), aad.len())?;

    let mut plaintext = Vec::with_capacity(ciphertext.len());
    // SAFETY: the reserved capacity covers ciphertext.len() bytes and
    // decrypt_raw() initializes all of them before the length is set.
    let expected_tag = unsafe {
        let tag = decrypt_raw(ciphertext, aad, key, nonce, plaintext.as_mut_ptr());
        plaintext.set_len(ciphertext.len());
        tag
    };

    if !ct_eq(tag, &expected_tag) {
        return Err(Error::AuthenticationFailed);
    }

    Ok(plaintext)
}

/// Decrypts ciphertext into a caller-provided buffer, avoiding allocation.
///
/// `plaintext` must be exactly `ciphertext.len()` bytes long.
/// If tag verification fails, the buffer is zeroed and an error is returned.
///
/// # Example
///
/// ```rust
/// use hiae::{encrypt_into, decrypt_into};
///
/// let key = [0u8; 32];
/// let nonce = [0u8; 16];
/// let mut ciphertext = [0u8; 14];
/// let tag = encrypt_into(b"secret message", b"", &key, &nonce, &mut ciphertext)?;
///
/// let mut plaintext = [0u8; 14];
/// decrypt_into(&ciphertext, &tag, b"", &key, &nonce, &mut plaintext)?;
/// assert_eq!(&plaintext, b"secret message");
/// # Ok::<(), hiae::Error>(())
/// ```
#[allow(unsafe_code)]
pub fn decrypt_into(
    ciphertext: &[u8],
    tag: &[u8; 16],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
    plaintext: &mut [u8],
) -> Result<()> {
    utils::validate_decrypt_params(ciphertext.len(), aad.len())?;
    if plaintext.len() != ciphertext.len() {
        return Err(Error::OutputBufferMismatch);
    }

    // SAFETY: plaintext is exactly ciphertext.len() bytes.
    let expected_tag = unsafe { decrypt_raw(ciphertext, aad, key, nonce, plaintext.as_mut_ptr()) };

    if !ct_eq(tag, &expected_tag) {
        plaintext.fill(0);
        return Err(Error::AuthenticationFailed);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let (ciphertext, tag) = encrypt(b"", b"", &key, &nonce).unwrap();
        assert!(ciphertext.is_empty());

        let decrypted = decrypt(&ciphertext, &tag, b"", &key, &nonce).unwrap();
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_all_message_lengths_roundtrip() {
        let key = [0x07u8; 32];
        let nonce = [0x0bu8; 16];
        let aad = b"header";
        let msg: Vec<u8> = (0..600).map(|i| i as u8).collect();

        for len in 0..msg.len() {
            let (ciphertext, tag) = encrypt(&msg[..len], aad, &key, &nonce).unwrap();
            let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce).unwrap();
            assert_eq!(decrypted, &msg[..len], "roundtrip failed at length {len}");
        }
    }

    #[test]
    fn test_authentication_failure() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];
        let plaintext = b"secret message";
        let aad = b"public header";

        let (ciphertext, mut tag) = encrypt(plaintext, aad, &key, &nonce).unwrap();
        tag[0] ^= 1;

        let result = decrypt(&ciphertext, &tag, aad, &key, &nonce);
        assert!(matches!(result, Err(Error::AuthenticationFailed)));
    }

    #[test]
    fn test_into_matches_allocating_api() {
        let key = [0x11u8; 32];
        let nonce = [0x22u8; 16];
        let aad = b"header";
        let msg: Vec<u8> = (0..600).map(|i| (i * 3) as u8).collect();

        for len in [0, 1, 15, 16, 17, 255, 256, 257, 300, 511, 512, 600] {
            let (expected_ct, expected_tag) = encrypt(&msg[..len], aad, &key, &nonce).unwrap();

            let mut ct = vec![0u8; len];
            let tag = encrypt_into(&msg[..len], aad, &key, &nonce, &mut ct).unwrap();
            assert_eq!(ct, expected_ct, "ciphertext mismatch at length {len}");
            assert_eq!(tag, expected_tag, "tag mismatch at length {len}");

            let mut pt = vec![0u8; len];
            decrypt_into(&ct, &tag, aad, &key, &nonce, &mut pt).unwrap();
            assert_eq!(pt, &msg[..len], "plaintext mismatch at length {len}");
        }
    }

    #[test]
    fn test_into_buffer_length_mismatch() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];
        let mut short = [0u8; 3];

        let result = encrypt_into(b"four", b"", &key, &nonce, &mut short);
        assert!(matches!(result, Err(Error::OutputBufferMismatch)));

        let result = decrypt_into(b"four", &[0u8; 16], b"", &key, &nonce, &mut short);
        assert!(matches!(result, Err(Error::OutputBufferMismatch)));
    }

    #[test]
    fn test_decrypt_into_zeroes_buffer_on_bad_tag() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];

        let mut ct = [0u8; 40];
        let mut tag = encrypt_into(&[0xaau8; 40], b"", &key, &nonce, &mut ct).unwrap();
        tag[0] ^= 1;

        let mut pt = [0x55u8; 40];
        let result = decrypt_into(&ct, &tag, b"", &key, &nonce, &mut pt);
        assert!(matches!(result, Err(Error::AuthenticationFailed)));
        assert_eq!(pt, [0u8; 40]);
    }

    #[test]
    fn test_corrupted_ciphertext_fails() {
        let key = [0u8; 32];
        let nonce = [0u8; 16];
        let plaintext = [0x55u8; 300];

        let (mut ciphertext, tag) = encrypt(&plaintext, b"", &key, &nonce).unwrap();
        ciphertext[299] ^= 1;

        let result = decrypt(&ciphertext, &tag, b"", &key, &nonce);
        assert!(matches!(result, Err(Error::AuthenticationFailed)));
    }
}
