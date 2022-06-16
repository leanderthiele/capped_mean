#include <torch/extension.h>

// forward declarations from the .cu file
torch::Tensor
capped_mean_forward_CU
    (const torch::Tensor &x, const torch::Tensor &N, bool keepdim=false);

torch::Tensor
capped_mean_backward_CU
    (const torch::Tensor &x, const torch::Tensor &N, const torch::Tensor &grad);

// and now the C++ calling code
torch::Tensor
capped_mean_forward
    (const torch::Tensor &x, const torch::Tensor &N, bool keepdim=false)
{
    return capped_mean_forward_CU(x, N, keepdim);
}

torch::Tensor
capped_mean_backward
    (const torch::Tensor &x, const torch::Tensor &N, const torch::Tensor &grad)
{
    return capped_mean_backward_CU(x, N, grad);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &capped_mean_forward, "capped mean forward (CUDA)");
    m.def("backward", &capped_mean_backward, "capped mean backward (CUDA)");
}
