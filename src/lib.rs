//! # HiAE - High-throughput Authenticated Encryption
//!
//! This crate provides an implementation of the HiAE (High-throughput Authenticated Encryption)
//! algorithm as specified in the IETF Internet-Draft.
//!
//! HiAE is designed for high-performance authenticated encryption with cross-platform efficiency,
//! particularly optimized for both ARM NEON and x86-64 AES-NI architectures.
//!
//! ## Features
//!
//! - **High Performance**: Leverages platform-specific SIMD instructions (ARM NEON, x86-64 AES-NI)
//! - **Security**: 256-bit keys, 128-bit nonces and tags, constant-time operations
//! - **Cross-Platform**: Optimized for both ARM and x86 architectures
//! - **Memory Safe**: Implemented in Rust with secure memory management
//! - **No-std Compatible**: Can be used in embedded environments
//!
//! ## Usage
//!
//! ```rust
//! use hiae::{encrypt, decrypt};
//!
//! let key = [0u8; 32];      // 256-bit key
//! let nonce = [0u8; 16];    // 128-bit nonce
//! let plaintext = b"Hello, world!";
//! let aad = b"additional data";
//!
//! // Encrypt
//! let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce)?;
//!
//! // Decrypt
//! let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce)?;
//! assert_eq!(decrypted, plaintext);
//! # Ok::<(), hiae::Error>(())
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![warn(missing_docs, rust_2018_idioms)]
#![deny(unsafe_code)]

extern crate alloc;

mod core;
mod error;
mod intrinsics;
mod utils;

#[cfg(test)]
mod tests;

pub use error::{Error, Result};

use alloc::vec::Vec;

/// Encrypts plaintext with associated data using HiAE.
///
/// # Arguments
///
/// * `plaintext` - The data to encrypt
/// * `aad` - Additional authenticated data (not encrypted, but authenticated)
/// * `key` - 256-bit encryption key
/// * `nonce` - 128-bit nonce (must be unique for each encryption with the same key)
///
/// # Returns
///
/// A tuple of (ciphertext, authentication_tag) on success, or an error.
///
/// # Security
///
/// - The nonce MUST NOT be reused with the same key
/// - The key MUST be randomly chosen from a uniform distribution
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
pub fn encrypt(
    plaintext: &[u8],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
) -> Result<(Vec<u8>, [u8; 16])> {
    core::encrypt(plaintext, aad, key, nonce)
}

/// Decrypts ciphertext and verifies the authentication tag.
///
/// # Arguments
///
/// * `ciphertext` - The encrypted data
/// * `tag` - 128-bit authentication tag
/// * `aad` - Additional authenticated data (must match encryption)
/// * `key` - 256-bit encryption key (must match encryption)
/// * `nonce` - 128-bit nonce (must match encryption)
///
/// # Returns
///
/// The decrypted plaintext on success, or an error if tag verification fails.
///
/// # Security
///
/// - If tag verification fails, no plaintext data is returned
/// - Tag comparison is performed in constant time
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
pub fn decrypt(
    ciphertext: &[u8],
    tag: &[u8; 16],
    aad: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 16],
) -> Result<Vec<u8>> {
    core::decrypt(ciphertext, tag, aad, key, nonce)
}
