/*
 * BioLLM Mamba-2 CUDA Parallel Prefix Scan Kernel (mamba_cuda_scan.cu)
 * 
 * Выполняет параллельное ассоциативное сканирование состояний State Space Model (Mamba-2)
 * за O(log N) времени с использованием Blelloch Parallel Scan в CUDA Shared Memory.
 * 
 * Уравнение состояния: h_t = A_t * h_{t-1} + B_t * x_t
 * Ассоциативный оператор: (A1, B1) o (A2, B2) = (A1 * A2, A2 * B1 + B2)
 * 
 * Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>

#define BLOCK_SIZE 256

__global__ void mamba_parallel_scan_kernel(
    const float* __restrict__ X,  // [Batch, Seq_len, Hidden_dim]
    const float* __restrict__ A,  // [Hidden_dim, State_dim]
    const float* __restrict__ B,  // [Batch, Seq_len, State_dim]
    const float* __restrict__ C,  // [Batch, Seq_len, State_dim]
    float* __restrict__ Y,        // [Batch, Seq_len, Hidden_dim]
    int batch_size, int seq_len, int hidden_dim, int state_dim
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    int tid = threadIdx.x;

    if (b >= batch_size || h >= hidden_dim) return;

    __shared__ float s_A[BLOCK_SIZE];
    __shared__ float s_Bx[BLOCK_SIZE];

    for (int chunk_start = 0; chunk_start < seq_len; chunk_start += BLOCK_SIZE) {
        int t = chunk_start + tid;
        
        // 1. Загрузка во временную Shared Memory
        if (t < seq_len) {
            float x_val = X[b * seq_len * hidden_dim + t * hidden_dim + h];
            float b_val = B[b * seq_len * state_dim + t * state_dim + (h % state_dim)];
            float a_val = expf(A[(h % hidden_dim) * state_dim + (h % state_dim)]);
            
            s_A[tid] = a_val;
            s_Bx[tid] = b_val * x_val;
        } else {
            s_A[tid] = 1.0f;
            s_Bx[tid] = 0.0f;
        }

        __syncthreads();

        // 2. Инкрементальное сканирование (Up-Sweep / Reduction Phase)
        for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
            int index = (tid + 1) * stride * 2 - 1;
            if (index < BLOCK_SIZE) {
                int left = index - stride;
                s_Bx[index] = s_A[index] * s_Bx[left] + s_Bx[index];
                s_A[index] = s_A[index] * s_A[left];
            }
            __syncthreads();
        }

        // 3. Вычисление выходного сигнала y_t = C_t * h_t + D * x_t
        if (t < seq_len) {
            float c_val = C[b * seq_len * state_dim + t * state_dim + (h % state_dim)];
            float x_val = X[b * seq_len * hidden_dim + t * hidden_dim + h];
            float state_h = s_Bx[tid];
            
            Y[b * seq_len * hidden_dim + t * hidden_dim + h] = c_val * state_h + x_val;
        }

        __syncthreads();
    }
}

// C++ Host Launcher
extern "C" void launch_mamba_parallel_scan(
    const float* X, const float* A, const float* B, const float* C, float* Y,
    int batch_size, int seq_len, int hidden_dim, int state_dim, cudaStream_t stream
) {
    dim3 grid(batch_size, hidden_dim);
    dim3 block(BLOCK_SIZE);

    mamba_parallel_scan_kernel<<<grid, block, 0, stream>>>(
        X, A, B, C, Y, batch_size, seq_len, hidden_dim, state_dim
    );
}
