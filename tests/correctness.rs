use asm_avx_simd::{equal, mul_avx512_asm, mul_avx512_asm_par, mul_avx512_block_par, mul_scalar, mul_scalar_par, Matrix};

#[test]
fn test_matrix_operations_correctness() {
    let size = 128;
    let a = Matrix::random(size, size);
    let b = Matrix::random(size, size);

    // Эталонный результат
    let scalar_result = mul_scalar(&a, &b);

    // Тестируем многоядерную скалярную версию
    let parallel_result = mul_scalar_par(&a, &b);
    assert!(equal(&scalar_result, &parallel_result, 1e-5));

    if is_x86_feature_detected!("avx512f") {
        // Тестируем AVX-512 версии
        let avx512_result = unsafe { mul_avx512_asm(&a, &b) };
        assert!(equal(&scalar_result, &avx512_result, 1e-3));

        let avx512_parallel_result = unsafe { mul_avx512_asm_par(&a, &b) };
        assert!(equal(&scalar_result, &avx512_parallel_result, 1e-3));
    }
}

#[test]
fn test_avx512_block_par_correctness() {
    if !is_x86_feature_detected!("avx512f") { return; }

    let n = 256;
    let a = Matrix::random(n, n);
    let b = Matrix::random(n, n);

    let reference_result = mul_scalar(&a, &b);
    let fast = unsafe { mul_avx512_block_par(&a, &b) };

    assert!(equal(&reference_result, &fast, 1e-3));
}

#[test]
fn test_avx512_availability() {
    println!("AVX-512F: {}", is_x86_feature_detected!("avx512f"));
    println!("AVX-512VL: {}", is_x86_feature_detected!("avx512vl"));
    println!("AVX-512BW: {}", is_x86_feature_detected!("avx512bw"));
}