/*
 * BioLLM Mamba-2 CUDA Extension Binding (mamba_cuda_extension.cpp)
 * 
 * Интерфейс PyBind11 для связи CUDA ядра mamba_cuda_scan.cu с PyTorch.
 * 
 * Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
 */

#include <torch/extension.h>
#include <vector>

extern "C" void launch_mamba_parallel_scan(
    const float* X, const float* A, const float* B, const float* C, float* Y,
    int batch_size, int seq_len, int hidden_dim, int state_dim, cudaStream_t stream
);

torch::Tensor mamba_parallel_scan_forward(
    torch::Tensor X,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
    TORCH_CHECK(X.is_cuda(), "Input X must be a CUDA tensor");
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "Input C must be a CUDA tensor");

    int batch_size = X.size(0);
    int seq_len = X.size(1);
    int hidden_dim = X.size(2);
    int state_dim = B.size(2);

    auto Y = torch::zeros_like(X);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_mamba_parallel_scan(
        X.data_ptr<float>(),
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        Y.data_ptr<float>(),
        batch_size, seq_len, hidden_dim, state_dim, stream
    );

    return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &mamba_parallel_scan_forward, "Mamba-2 Parallel Scan CUDA forward");
}
