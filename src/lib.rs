#![feature(portable_simd)]
#![allow(clippy::too_many_arguments)]

use std::arch::asm;
use std::arch::x86_64::*;
use rayon::prelude::*;

/// Параметры матрицы по умолчанию
pub const MATRIX_SIZE: usize = 2048;
const LANES: usize = 16;                // 512 бит ÷ 32 бит = 16
const BLOCK_M: usize = 64;    // строк в блоке (m‑тайл)
const BLOCK_K: usize = 128;   // глубина k‑тайла
const PREFETCH_DIST: isize = 32;    // на 32 элемента k вперёд (~200 тактов)

/// 64-байтное выравнивание для ZMM-загрузок
#[repr(align(64))]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut m = Self::new(rows, cols);
        for i in 0..m.data.len() { m.data[i] = (i as f32 * 0.37).sin(); }
        m
    }
    #[inline] pub fn get (&self, r:usize, c:usize)->f32 { self.data[r*self.cols+c] }
    #[inline] pub fn set (&mut self,r:usize,c:usize,v:f32){ self.data[r*self.cols+c]=v }
}

/* ------------------------------------------------------------------------- */
/* 1. Скалярная функция — 1 ядро                                             */
/* ------------------------------------------------------------------------- */
pub fn mul_scalar(a:&Matrix,b:&Matrix)->Matrix{
    assert_eq!(a.cols,b.rows);
    let mut c=Matrix::new(a.rows,b.cols);
    for i in 0..a.rows{
        for j in 0..b.cols{
            let mut sum=0.0;
            for k in 0..a.cols{
                sum+=a.get(i,k)*b.get(k,j);
            }
            c.set(i,j,sum);
        }
    }
    c
}

/* ------------------------------------------------------------------------- */
/* 2. Скаляр + Rayon — 8 ядер / 16 потоков                                   */
/* ------------------------------------------------------------------------- */
pub fn mul_scalar_par(a:&Matrix,b:&Matrix)->Matrix{
    assert_eq!(a.cols,b.rows);
    let mut c=Matrix::new(a.rows,b.cols);
    c.data.par_chunks_mut(b.cols)
        .enumerate()
        .for_each(|(row,chunk)|{
            for j in 0..b.cols{
                let mut sum=0.0;
                for k in 0..a.cols{ sum+=a.get(row,k)*b.get(k,j);}
                chunk[j]=sum;
            }
        });
    c
}

/* ------------------------------------------------------------------------- */
/* 3. ASM + AVX-512 — 1 ядро                                                 */
/* ------------------------------------------------------------------------- */
#[target_feature(enable="avx512f,avx512vl")]
pub unsafe fn mul_avx512_asm(a:&Matrix,b:&Matrix)->Matrix{
    assert_eq!(a.cols,b.rows);
    let mut c=Matrix::new(a.rows,b.cols);

    for i in 0..a.rows{
        for j in (0..b.cols).step_by(LANES){
            // обнуляем регистр
            let mut acc:__m512;
            asm!("vpxord {acc}, {acc}, {acc}",
                 acc=out(zmm_reg) acc,
                 options(pure, nomem, nostack));

            for k in 0..a.cols{
                let scalar = a.get(i,k);

                // BROADCAST через указатель (&scalar)
                let broadcast:__m512;
                unsafe{
                    asm!(
                        "vbroadcastss {out}, DWORD PTR [{src}]",
                        out = out(zmm_reg) broadcast,
                        src = in(reg) &scalar,
                        options(pure, readonly, nostack),
                    );
                }

                let b_ptr = &b.data[k*b.cols+j] as *const f32;

                let b_vec:__m512;
                unsafe{
                    asm!(
                        "vmovups {out}, [{src}]",
                        out = out(zmm_reg) b_vec,
                        src = in(reg)     b_ptr,
                        options(pure, readonly, nostack));
                }

                // FMA
                asm!(
                    "vfmadd231ps {acc}, {a_vec}, {b_vec}",
                    acc   = inout(zmm_reg) acc,
                    a_vec = in(zmm_reg)   broadcast,
                    b_vec = in(zmm_reg)   b_vec,
                    options(pure, nomem, nostack)
                );
            }

            let dst = &mut c.data[i*c.cols+j] as *mut f32;
            asm!("vmovups [{ptr}], {acc}",
                 ptr=in(reg) dst,
                 acc=in(zmm_reg) acc,
                 options(nostack)
            );
        }
    }
    c
}

/* ------------------------------------------------------------------------- */
/* 4. ASM + AVX-512 + Rayon — 8 ядер / 16 потоков                            */
/* ------------------------------------------------------------------------- */
#[target_feature(enable="avx512f,avx512vl")]
pub fn mul_avx512_asm_par(a:&Matrix,b:&Matrix)->Matrix{
    assert_eq!(a.cols,b.rows);
    if !is_x86_feature_detected!("avx512f"){ panic!("AVX-512 не поддерживается"); }
    let mut c=Matrix::new(a.rows,b.cols);

    c.data.par_chunks_mut(b.cols)
        .enumerate()
        .for_each(|(row,chunk)|{
            unsafe{
                for j in (0..b.cols).step_by(LANES){
                    let mut acc:__m512;
                    asm!("vpxord {acc}, {acc}, {acc}",
                         acc=out(zmm_reg) acc,
                         options(pure, nomem, nostack));

                    for k in 0..a.cols{
                        let scalar=a.get(row,k);
                        let bptr=&b.data[k*b.cols+j] as *const f32;

                        let a_vec:__m512;
                        asm!("vbroadcastss {v}, DWORD PTR [{p}]",
                             v=out(zmm_reg) a_vec,
                             p=in(reg)&scalar,
                             options(pure, readonly, nostack));

                        let b_vec:__m512;
                        asm!("vmovups {v}, [{p}]",v=out(zmm_reg)b_vec,p=in(reg)bptr,
                             options(pure, readonly, nostack));

                        asm!("vfmadd231ps {acc}, {a_vec}, {b_vec}",
                             acc=inout(zmm_reg)acc,
                             a_vec=in(zmm_reg)a_vec,
                             b_vec=in(zmm_reg)b_vec,
                             options(pure, nomem, nostack));
                    }
                    let dst=&mut chunk[j] as *mut f32;
                    asm!("vmovups [{p}], {acc}",
                         p=in(reg) dst,
                         acc=in(zmm_reg) acc,
                         options(nostack));
                }
            }
        });
    c
}


/// Блочный k-тайлинг + AVX-512 + Rayon
#[target_feature(enable = "avx512f,avx512vl")]
pub unsafe fn mul_avx512_block_par(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "A.cols must equal B.rows");

    let (m, n, k) = (a.rows, b.cols, a.cols);
    let mut c = Matrix::new(m, n);

    // каждые BLOCK_M строк C обрабатываются независимым потоком
    c.data
        .par_chunks_mut(BLOCK_M * n)
        .enumerate()
        .for_each(|(blk_idx, c_chunk)| {
            let row_start = blk_idx * BLOCK_M;
            let row_end   = (row_start + BLOCK_M).min(m);
            let big       = n >= 3072;                    // ← крупная матрица?

            for kk in (0..k).step_by(BLOCK_K) {
                let kend = (kk + BLOCK_K).min(k);

                for i in row_start..row_end {
                    let off = (i - row_start) * n;

                    for j in (0..n).step_by(LANES) {
                        let mut acc = _mm512_setzero_ps();

                        // ---------- главный k‑цикл ----------
                        let mut kx = kk;
                        while kx + 4 <= kend {
                            // префетч будущей панели B (только для big)
                            if big && (kx as isize + PREFETCH_DIST) < k as isize {
                                let pf_ptr = b.data.as_ptr()
                                    .add((kx + PREFETCH_DIST as usize) * n + j) as *const i8;
                                _mm_prefetch::<{ _MM_HINT_T1 }>(pf_ptr);
                            }

                            macro_rules! fma_step {
                                ($ofs:expr) => {{
                                    let a_val = _mm512_set1_ps(
                                        *a.data.get_unchecked(i * k + kx + $ofs)
                                    );
                                    let b_vec = _mm512_loadu_ps(
                                        b.data.as_ptr().add((kx + $ofs) * n + j)
                                    );
                                    acc = _mm512_fmadd_ps(a_val, b_vec, acc);
                                }};
                            }
                            fma_step!(0);
                            fma_step!(1);
                            fma_step!(2);
                            fma_step!(3);
                            kx += 4;
                        }
                        while kx < kend {
                            let a_val = _mm512_set1_ps(*a.data.get_unchecked(i * k + kx));
                            let b_vec = _mm512_loadu_ps(
                                b.data.as_ptr().add(kx * n + j)
                            );
                            acc = _mm512_fmadd_ps(a_val, b_vec, acc);
                            kx += 1;
                        }

                        // ---------- запись результата ----------
                        if j + LANES <= n {
                            if big {
                                _mm512_stream_ps(
                                    c_chunk[off + j..].as_mut_ptr(),
                                    acc,
                                );
                            } else {
                                _mm512_storeu_ps(
                                    c_chunk[off + j..].as_mut_ptr(),
                                    acc,
                                );
                            }
                        } else {
                            // хвост < 16 элементов
                            let mut tmp = [0f32; LANES];
                            _mm512_storeu_ps(tmp.as_mut_ptr(), acc);
                            for t in 0..(n - j) {
                                c_chunk[off + j + t] = tmp[t];
                            }
                        }
                    }
                }
            }
        });

    c
}


/* ------------------------------------------------------------------------- */
/* Утилиты                                                                   */
/* ------------------------------------------------------------------------- */
pub fn equal(a:&Matrix,b:&Matrix,eps:f32)->bool{
    if a.rows!=b.rows||a.cols!=b.cols{ return false }
    for i in 0..a.data.len(){
        if (a.data[i]-b.data[i]).abs()>eps{ return false }
    }
    true
}

/// Настройка глобального пула Rayon (16 потоков)
pub fn init_pool(){ let _=rayon::ThreadPoolBuilder::new()
        .num_threads(16).build_global(); }
