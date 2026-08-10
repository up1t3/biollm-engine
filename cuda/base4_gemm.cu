/*
 * BioLLM Base-4 CUDA GEMM Kernel (base4_gemm.cu)
 * 
 * Выполняет высокоскоростное умножение матриц Y = X * W^T с распаковкой 2-битных весов Base-4 DNA
 * (A=00, C=01, G=10, T=11) прямо в быстрой памяти L1 / Shared Memory (SRAM) GPU.
 * 
 * Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

// CUDA Kernel: Base-4 Dequantization + GEMM Accumulation in Shared Memory
__global__ void base4_gemm_kernel(
    const float* __restrict__ A,          // [Batch, K] - Input Activations
    const uint8_t* __restrict__ B_packed, // [N, K/4]  - 2-bit Packed Base-4 Weight Matrix
    float* __restrict__ C,                // [Batch, N] - Output Activations
    const float* __restrict__ scales,     // [N]        - Per-channel Quantization Scales
    const float* __restrict__ min_vals,   // [N]        - Per-channel Quantization Min Values
    int M, int N, int K
) {
    int row = blockIdx.y * TILE_M + threadIdx.y;
    int col = blockIdx.x * TILE_N + threadIdx.x;

    __shared__ float s_A[TILE_M][TILE_K];
    __shared__ float s_B[TILE_N][TILE_K];

    float accum = 0.0f;

    for (int k_tile = 0; k_tile < (K + TILE_K - 1) / TILE_K; ++k_tile) {
        // 1. Загрузка активаций A в Shared Memory
        if (row < M && (k_tile * TILE_K + threadIdx.x) < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + k_tile * TILE_K + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 2. Распаковка 2-битных нуклеотидов B_packed в Shared Memory
        if (col < N && (k_tile * TILE_K + threadIdx.y) < K) {
            int k_global = k_tile * TILE_K + threadIdx.y;
            int byte_idx = col * (K / 4) + (k_global / 4);
            int bit_shift = (3 - (k_global % 4)) * 2;
            
            uint8_t packed_byte = B_packed[byte_idx];
            uint8_t code = (packed_byte >> bit_shift) & 0x03;
            
            // Восстановление вещественного значения: dequant = code / scale + min_val
            float scale = scales[col];
            float min_val = min_vals[col];
            s_B[threadIdx.x][threadIdx.y] = (float)code / scale + min_val;
        } else {
            s_B[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // 3. Вычисление скалярного произведения блока
        for (int k = 0; k < TILE_K; ++k) {
            accum += s_A[threadIdx.y][k] * s_B[threadIdx.x][k];
        }

        __syncthreads();
    }

    // 4. Запись результата C
    if (row < M && col < N) {
        C[row * N + col] = accum;
    }
}

// C++ Host Wrapper Function
extern "C" void launch_base4_gemm(
    const float* A, const uint8_t* B_packed, float* C,
    const float* scales, const float* min_vals,
    int M, int N, int K, cudaStream_t stream
) {
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(TILE_N, TILE_M);

    base4_gemm_kernel<<<grid, block, 0, stream>>>(
        A, B_packed, C, scales, min_vals, M, N, K
    );
}
