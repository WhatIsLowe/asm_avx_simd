use criterion::{ criterion_group, criterion_main, Criterion, BenchmarkId };
use asm_avx_simd::{init_pool, mul_avx512_asm, mul_avx512_asm_par, mul_avx512_block_par, mul_scalar, mul_scalar_par, Matrix};
use std::hint::black_box;

fn benchmark_all_implementations(c: &mut Criterion) {
    // Настройка пула потоков
    init_pool();

    let sizes = [256, 512, 1024, 2048, 4096, 8192];
    // let sizes = [2048,];

    for &size in &sizes {
        let mut group = c.benchmark_group(format!("Matrix {}x{}", size, size));
        group.sample_size(10);      // Кол-во сэмплов

        // Создание тестовых матриц
        let a = Matrix::random(size, size);
        let b = Matrix::random(size, size);

        if size <= 512 {
            // Скалярная версия, 1 ядро
            group.bench_with_input(
                BenchmarkId::new("Scalar_SingleCore", size),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        black_box(mul_scalar(black_box(a), black_box(b)))
                    })
                },
            );
        }

        if size <= 1024 {
            // Скалярная версия, многоядерная
            group.bench_with_input(
                BenchmarkId::new("Scalar_MultiCore", size),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| {
                        black_box(mul_scalar_par(black_box(a), black_box(b)))
                    })
                },
            );

            if is_x86_feature_detected!("avx512f") {
                // AVX-512 + ASM, 1 ядро
                group.bench_with_input(
                    BenchmarkId::new("AVX512_ASM_SingleCore", size),
                    &(&a, &b),
                    |bench, (a, b)| {
                        bench.iter(|| unsafe {
                            black_box(mul_avx512_asm(black_box(a), black_box(b)))
                        })
                    },
                );
            }
        }

        if is_x86_feature_detected!("avx512f") {

            // AVX-512 + ASM, многоядерная
            group.bench_with_input(
                BenchmarkId::new("AVX512_ASM_MultiCore", size),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| unsafe {
                        black_box(mul_avx512_asm_par(black_box(a), black_box(b)))
                    })
                },
            );

            // Блочная AVX-512, многоядерная
            group.bench_with_input(
                BenchmarkId::new("AVX512_BLOCK_MultiCore", size),
                &(&a, &b),
                |bench, (a, b)| {
                    bench.iter(|| unsafe {
                        black_box(mul_avx512_block_par(black_box(a), black_box(b)))
                    })
                },
            );

        } else {
            println!("AVX-512 is not supported");
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_all_implementations);
criterion_main!(benches);