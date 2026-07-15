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
//! - **Security**: 256-bit keys, 128-bit nonces and tags
//! - **Cross-Platform**: Optimized for both ARM and x86 architectures
//! - **No-std Compatible**: Can be used in embedded environments
//!
//! On the hardware-accelerated backends, processing is constant time.
//! The portable fallback used on other targets relies on table lookups and is
//! not constant time.
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

mod backend;
mod core;
mod error;
mod utils;

#[cfg(test)]
mod tests;

pub use error::{Error, Result};

pub use crate::core::{decrypt, decrypt_into, encrypt, encrypt_into};
