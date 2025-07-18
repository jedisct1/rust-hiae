# HiAE - High-throughput Authenticated Encryption

A Rust implementation of the HiAE (High-throughput Authenticated Encryption) algorithm, providing authenticated encryption with associated data (AEAD) optimized for high performance across different architectures.

## Features

- **High Performance**: Leverages platform-specific SIMD instructions (ARM NEON, x86-64 AES-NI)
- **Security**: 256-bit keys, 128-bit nonces and tags, constant-time operations
- **Cross-Platform**: Optimized for both ARM and x86 architectures with fallback implementations
- **Memory Safe**: Implemented in Rust with automatic memory zeroing and no unsafe code
- **No-std Compatible**: Can be used in embedded environments

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
hiae = "0.1.0"
```

Basic usage:

```rust
use hiae::{encrypt, decrypt};

let key = [0u8; 32];      // 256-bit key
let nonce = [0u8; 16];    // 128-bit nonce
let plaintext = b"Hello, world!";
let aad = b"additional data";

// Encrypt
let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce)?;

// Decrypt
let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce)?;
assert_eq!(decrypted, plaintext);
```

## Performance Optimization

### Enabling Native CPU Features

To get maximum performance, compile with native CPU optimizations enabled. This allows the library to automatically detect and use the best available SIMD instructions for your CPU.

#### Option 1: Environment Variable (Recommended)

Set the `RUSTFLAGS` environment variable before building:

```bash
# For maximum performance on your current CPU
export RUSTFLAGS="-C target-cpu=native"
cargo build --release

# Or run directly
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

#### Option 2: Cargo Configuration

Create or edit `.cargo/config.toml` in your project root:

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

Then build normally:

```bash
cargo build --release
```

#### Option 3: Target-Specific Flags

For specific CPU features, you can enable them explicitly:

```bash
# For x86-64 with AES-NI
RUSTFLAGS="-C target-feature=+aes,+pclmul" cargo build --release

# For ARM with NEON
RUSTFLAGS="-C target-feature=+neon" cargo build --release
```

### Architecture-Specific Features

The library includes optional feature flags for explicit architecture support:

```bash
# Build with x86-64 AES-NI support
cargo build --features aes-ni --release

# Build with ARM NEON support  
cargo build --features neon --release

# Build without std for embedded use
cargo build --no-default-features --release
```

### Performance Verification

Run the benchmarks to verify performance gains:

```bash
cargo bench
```

The benchmarks will show throughput improvements when CPU-specific optimizations are enabled.

## Building and Testing

### Standard Build Commands

```bash
# Development build
cargo build

# Optimized release build
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Generate documentation
cargo doc --open
```

### Running Examples

```bash
# Basic usage example
cargo run --example basic_usage

# With optimizations
RUSTFLAGS="-C target-cpu=native" cargo run --example basic_usage --release
```

### Benchmarking

```bash
# Run performance benchmarks
cargo bench

# With native optimizations
RUSTFLAGS="-C target-cpu=native" cargo bench
```

## Algorithm Specification

HiAE is based on the [IETF Internet-Draft](https://github.com/hiae-aead/draft-pham-hiae) and provides:

- **Key Size**: 256 bits (32 bytes)
- **Nonce Size**: 128 bits (16 bytes) 
- **Tag Size**: 128 bits (16 bytes)
- **Maximum Message Length**: 2^61 - 1 bytes
- **Block Size**: 128 bits (16 bytes)

## Security Notes

- **Nonce Reuse**: Never reuse a nonce with the same key
- **Key Generation**: Use cryptographically secure random number generators
- **Constant Time**: Tag verification is performed in constant time to prevent timing attacks
- **Memory Safety**: All sensitive data is automatically zeroed after use

## Supported Platforms

The library automatically detects and uses the best available implementation:

| Platform | SIMD Instructions | Performance |
|----------|------------------|-------------|
| x86-64 | AES-NI + PCLMUL | Highest |
| ARM64 | NEON + AES | Highest |
| Other | Portable fallback | Good |

## API Documentation

For detailed API documentation, run:

```bash
cargo doc --open
```

## Examples

See the `examples/` directory for comprehensive usage examples:

- `basic_usage.rs` - Encryption, decryption, and error handling

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.

## Contributing

This implementation follows the HiAE specification. For bugs or improvements, please open an issue or pull request.