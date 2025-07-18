use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hiae::{decrypt, encrypt};
use std::hint::black_box;

/// Generate test data of the specified size
fn generate_test_data(size: usize) -> (Vec<u8>, Vec<u8>, [u8; 32], [u8; 16]) {
    let plaintext = vec![0x42u8; size];
    let aad = vec![0x41u8; 32]; // Fixed AAD size
    let key = [0x01u8; 32];
    let nonce = [0x02u8; 16];
    (plaintext, aad, key, nonce)
}

/// Print CPU feature detection information
fn print_cpu_features() {
    println!("=== CPU Feature Detection ===");

    #[cfg(target_arch = "aarch64")]
    {
        let neon = std::arch::is_aarch64_feature_detected!("neon");
        let aes = std::arch::is_aarch64_feature_detected!("aes");
        println!("Architecture: ARM64/AArch64");
        println!("NEON support: {}", if neon { "✓" } else { "✗" });
        println!("AES Crypto Extensions: {}", if aes { "✓" } else { "✗" });
        println!(
            "Hardware acceleration: {}",
            if neon && aes { "ENABLED" } else { "DISABLED" }
        );
    }

    #[cfg(target_arch = "x86_64")]
    {
        let aes = std::arch::is_x86_feature_detected!("aes");
        let sse41 = std::arch::is_x86_feature_detected!("sse4.1");
        let sse42 = std::arch::is_x86_feature_detected!("sse4.2");
        println!("Architecture: x86_64");
        println!("AES-NI support: {}", if aes { "✓" } else { "✗" });
        println!("SSE4.1 support: {}", if sse41 { "✓" } else { "✗" });
        println!("SSE4.2 support: {}", if sse42 { "✓" } else { "✗" });
        println!(
            "Hardware acceleration: {}",
            if aes { "ENABLED" } else { "DISABLED" }
        );
    }

    println!("==============================\n");
}

/// Benchmark encryption performance across different data sizes
fn bench_encrypt_sizes(c: &mut Criterion) {
    print_cpu_features();
    let mut group = c.benchmark_group("encrypt_throughput");

    // Test sizes from 64 bytes to 1MB
    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes {
        let (plaintext, aad, key, nonce) = generate_test_data(size);

        group.throughput(Throughput::Elements((size as u64) * 8));
        group.bench_with_input(BenchmarkId::new("encrypt", size), &size, |b, _| {
            b.iter(|| {
                let result = encrypt(
                    black_box(&plaintext),
                    black_box(&aad),
                    black_box(&key),
                    black_box(&nonce),
                );
                black_box(result).unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark decryption performance across different data sizes
fn bench_decrypt_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("decrypt_throughput");

    // Test sizes from 64 bytes to 1MB
    let sizes = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576];

    for size in sizes {
        let (plaintext, aad, key, nonce) = generate_test_data(size);
        let (ciphertext, tag) = encrypt(&plaintext, &aad, &key, &nonce).unwrap();

        group.throughput(Throughput::Elements((size as u64) * 8));
        group.bench_with_input(BenchmarkId::new("decrypt", size), &size, |b, _| {
            b.iter(|| {
                let result = decrypt(
                    black_box(&ciphertext),
                    black_box(&tag),
                    black_box(&aad),
                    black_box(&key),
                    black_box(&nonce),
                );
                black_box(result).unwrap()
            });
        });
    }
    group.finish();
}

/// Benchmark round-trip (encrypt + decrypt) performance
fn bench_roundtrip_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip_throughput");

    // Test sizes for round-trip operations
    let sizes = [64, 1024, 16384, 262144];

    for size in sizes {
        let (plaintext, aad, key, nonce) = generate_test_data(size);

        group.throughput(Throughput::Elements((size as u64) * 8));
        group.bench_with_input(BenchmarkId::new("encrypt_decrypt", size), &size, |b, _| {
            b.iter(|| {
                let (ciphertext, tag) = encrypt(
                    black_box(&plaintext),
                    black_box(&aad),
                    black_box(&key),
                    black_box(&nonce),
                )
                .unwrap();

                let decrypted = decrypt(
                    black_box(&ciphertext),
                    black_box(&tag),
                    black_box(&aad),
                    black_box(&key),
                    black_box(&nonce),
                )
                .unwrap();

                black_box(decrypted)
            });
        });
    }
    group.finish();
}

/// Benchmark with varying AAD sizes
fn bench_aad_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("aad_sizes");

    let plaintext = vec![0x42u8; 1024]; // Fixed plaintext size
    let key = [0x01u8; 32];
    let nonce = [0x02u8; 16];

    // Test different AAD sizes
    let aad_sizes = [0, 16, 64, 256, 1024, 4096];

    for aad_size in aad_sizes {
        let aad = vec![0x41u8; aad_size];

        group.bench_with_input(
            BenchmarkId::new("encrypt_with_aad", aad_size),
            &aad_size,
            |b, _| {
                b.iter(|| {
                    let result = encrypt(
                        black_box(&plaintext),
                        black_box(&aad),
                        black_box(&key),
                        black_box(&nonce),
                    );
                    black_box(result).unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark key and nonce setup overhead
fn bench_setup_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("setup_overhead");

    let plaintext = vec![0x42u8; 64]; // Small plaintext to isolate setup cost
    let aad = vec![0x41u8; 16];
    let key = [0x01u8; 32];
    let nonce = [0x02u8; 16];

    group.bench_function("encrypt_64_bytes", |b| {
        b.iter(|| {
            let result = encrypt(
                black_box(&plaintext),
                black_box(&aad),
                black_box(&key),
                black_box(&nonce),
            );
            black_box(result).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_encrypt_sizes,
    bench_decrypt_sizes,
    bench_roundtrip_sizes,
    bench_aad_sizes,
    bench_setup_overhead
);
criterion_main!(benches);
