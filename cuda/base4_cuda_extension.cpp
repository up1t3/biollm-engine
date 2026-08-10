/*
 * BioLLM PyTorch C++ / CUDA Binding Extension (base4_cuda_extension.cpp)
 * 
 * Связывает CUDA-ядро base4_gemm.cu с макетом вызова PyTorch.
 * 
 * Автор: Vladimir Popov <up1t3r@gmail.com> & Antigravity AI
 */

#include <torch/extension.h>
#include <vector>

// Объявление функции C++ host wrapper из base4_gemm.cu
extern "C" void launch_base4_gemm(
    const float* A, const uint8_t* B_packed, float* C,
    const float* scales, const float* min_vals,
    int M, int N, int K, cudaStream_t stream
);

torch::Tensor base4_gemm_forward(
    torch::Tensor A,
    torch::Tensor B_packed,
    torch::Tensor scales,
    torch::Tensor min_vals
) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B_packed.is_cuda(), "Input B_packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "Scales must be a CUDA tensor");
    TORCH_CHECK(min_vals.is_cuda(), "Min_vals must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B_packed.size(0);

    auto C = torch::zeros({M, N}, A.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    launch_base4_gemm(
        A.data_ptr<float>(),
        B_packed.data_ptr<uint8_t>(),
        C.data_ptr<float>(),
        scales.data_ptr<float>(),
        min_vals.data_ptr<float>(),
        M, N, K, stream
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &base4_gemm_forward, "Base-4 2-bit DNA CUDA GEMM forward (CUDA)");
}
