#include <torch/extension.h>
    
#include "core.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &capped_mean_forward, "capped mean forward (CPU)");
    m.def("backward", &capped_mean_backward, "capped mean backward (CPU)");
}
