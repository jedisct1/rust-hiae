//! Error types for HiAE operations.

use core::fmt;

/// Result type alias for HiAE operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors that can occur during HiAE operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Plaintext too long (maximum 2^61 - 1 bytes).
    PlaintextTooLong,

    /// Associated data too long (maximum 2^61 - 1 bytes).
    AssociatedDataTooLong,

    /// Ciphertext too long.
    CiphertextTooLong,

    /// Output buffer length does not match the input length.
    OutputBufferMismatch,

    /// Authentication tag verification failed.
    AuthenticationFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::PlaintextTooLong => write!(f, "plaintext too long (maximum 2^61 - 1 bytes)"),
            Error::AssociatedDataTooLong => {
                write!(f, "associated data too long (maximum 2^61 - 1 bytes)")
            }
            Error::CiphertextTooLong => write!(f, "ciphertext too long"),
            Error::OutputBufferMismatch => {
                write!(f, "output buffer length does not match the input length")
            }
            Error::AuthenticationFailed => write!(f, "authentication tag verification failed"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}
