//! Basic usage example for HiAE authenticated encryption.

use hiae::{decrypt, encrypt, Error};

fn main() -> Result<(), Error> {
    println!("HiAE Basic Usage Example");
    println!("========================");

    // Example 1: Basic encryption and decryption
    basic_example()?;

    // Example 2: Handling different input sizes
    size_examples()?;

    // Example 3: Error handling
    error_handling_example()?;

    Ok(())
}

fn basic_example() -> Result<(), Error> {
    println!("\n1. Basic Encryption/Decryption:");

    let key = [0x01; 32]; // 256-bit key
    let nonce = [0x02; 16]; // 128-bit nonce
    let plaintext = b"Hello, HiAE! This is a secret message.";
    let aad = b"public header";

    // Encrypt
    let (ciphertext, tag) = encrypt(plaintext, aad, &key, &nonce)?;
    println!("  Plaintext: {:?}", String::from_utf8_lossy(plaintext));
    println!("  Ciphertext: {} bytes", ciphertext.len());
    println!("  Tag: {:02x?}", &tag[..8]); // Show first 8 bytes

    // Decrypt
    let decrypted = decrypt(&ciphertext, &tag, aad, &key, &nonce)?;
    println!("  Decrypted: {:?}", String::from_utf8_lossy(&decrypted));

    assert_eq!(decrypted, plaintext);
    println!("  ✓ Encryption/decryption successful!");

    Ok(())
}

fn size_examples() -> Result<(), Error> {
    println!("\n2. Different Input Sizes:");

    let key = [0x03; 32];
    let nonce = [0x04; 16];

    // Empty message
    let (ct, tag) = encrypt(b"", b"just aad", &key, &nonce)?;
    let pt = decrypt(&ct, &tag, b"just aad", &key, &nonce)?;
    println!("  Empty message: {} bytes -> {} bytes", 0, pt.len());
    assert!(pt.is_empty());

    // Single byte
    let (ct, tag) = encrypt(b"A", b"", &key, &nonce)?;
    let pt = decrypt(&ct, &tag, b"", &key, &nonce)?;
    println!("  Single byte: {} bytes -> {} bytes", 1, pt.len());
    assert_eq!(pt, b"A");

    // Large message (multiple blocks)
    let large_msg = vec![0x42u8; 1000];
    let (ct, tag) = encrypt(&large_msg, b"large message", &key, &nonce)?;
    let pt = decrypt(&ct, &tag, b"large message", &key, &nonce)?;
    println!(
        "  Large message: {} bytes -> {} bytes",
        large_msg.len(),
        pt.len()
    );
    assert_eq!(pt, large_msg);

    println!("  ✓ All size tests passed!");
    Ok(())
}

fn error_handling_example() -> Result<(), Error> {
    println!("\n3. Error Handling:");

    let key = [0x05; 32];
    let nonce = [0x06; 16];
    let plaintext = b"secret data";
    let aad = b"header";

    // Encrypt normally
    let (ciphertext, mut tag) = encrypt(plaintext, aad, &key, &nonce)?;

    // Test authentication failure with corrupted tag
    tag[0] ^= 1; // Flip one bit
    match decrypt(&ciphertext, &tag, aad, &key, &nonce) {
        Ok(_) => println!("  ✗ Should have failed!"),
        Err(Error::AuthenticationFailed) => {
            println!("  ✓ Authentication failure detected correctly");
        }
        Err(e) => println!("  ✗ Unexpected error: {e}"),
    }

    // Test with wrong AAD
    tag[0] ^= 1; // Fix the tag
    match decrypt(&ciphertext, &tag, b"wrong header", &key, &nonce) {
        Ok(_) => println!("  ✗ Should have failed!"),
        Err(Error::AuthenticationFailed) => {
            println!("  ✓ Wrong AAD detected correctly");
        }
        Err(e) => println!("  ✗ Unexpected error: {e}"),
    }

    // Test with wrong key
    let wrong_key = [0x99; 32];
    match decrypt(&ciphertext, &tag, aad, &wrong_key, &nonce) {
        Ok(_) => println!("  ✗ Should have failed!"),
        Err(Error::AuthenticationFailed) => {
            println!("  ✓ Wrong key detected correctly");
        }
        Err(e) => println!("  ✗ Unexpected error: {e}"),
    }

    println!("  ✓ All error handling tests passed!");
    Ok(())
}
