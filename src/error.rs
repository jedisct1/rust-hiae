//! Error types for HiAE operations.

use core::fmt;

/// Result type alias for HiAE operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during HiAE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Invalid key length (must be 32 bytes).
    InvalidKeyLength,

    /// Invalid nonce length (must be 16 bytes).
    InvalidNonceLength,

    /// Invalid tag length (must be 16 bytes).
    InvalidTagLength,

    /// Plaintext too long (maximum 2^61 - 1 bytes).
    PlaintextTooLong,

    /// Associated data too long (maximum 2^61 - 1 bytes).
    AssociatedDataTooLong,

    /// Ciphertext too long.
    CiphertextTooLong,

    /// Authentication tag verification failed.
    AuthenticationFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidKeyLength => write!(f, "invalid key length (must be 32 bytes)"),
            Error::InvalidNonceLength => write!(f, "invalid nonce length (must be 16 bytes)"),
            Error::InvalidTagLength => write!(f, "invalid tag length (must be 16 bytes)"),
            Error::PlaintextTooLong => write!(f, "plaintext too long (maximum 2^61 - 1 bytes)"),
            Error::AssociatedDataTooLong => {
                write!(f, "associated data too long (maximum 2^61 - 1 bytes)")
            }
            Error::CiphertextTooLong => write!(f, "ciphertext too long"),
            Error::AuthenticationFailed => write!(f, "authentication tag verification failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
