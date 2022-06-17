/* This header implements the core functionality.
 *
 * Define CAPPED_MEAN_CUDA before including to get the CUDA version.
 */

#include <algorithm>
#include <stdint.h>

#include <torch/extension.h>

#include "atom.h"
#include "shapes.h"

#ifdef CAPPED_MEAN_CUDA
template<Mode mode, typename TN, typename Tval>
__global__
static void
capped_mean_kernel
    (int64_t d1, int64_t d2, int64_t d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    int64_t idx1 = blockIdx.x,
            idx3 = threadIdx.x;

    capped_mean_atom<mode, TN, Tval>(idx1, idx3,
                                     d1, d2, d3,
                                     x, N, y);
}

// need to instantiate for all possible types
#define INSTANTIATE_KERNELS(TN, Tval)              \
    template __global__ void                       \
    capped_mean_kernel <FORWARD, TN, Tval>         \
        (int64_t, int64_t, int64_t,                \
         const Tval *, const TN *, Tval *);        \
                                                   \
    template __global__ void                       \
    capped_mean_kernel <BACKWARD, TN, Tval>        \
        (int64_t, int64_t, int64_t,                \
         const Tval *, const TN *, Tval *)

// at the moment, we don't do this for that many types, can always
// add if something comes up
INSTANTIATE_KERNELS(int, float);
INSTANTIATE_KERNELS(int64_t, float);
INSTANTIATE_KERNELS(int16_t, float);
INSTANTIATE_KERNELS(int, double);
INSTANTIATE_KERNELS(int64_t, double);
INSTANTIATE_KERNELS(int16_t, double);

#undef INSTANTIATE_KERNELS

#endif // CAPPED_MEAN_CUDA

template<Mode mode, typename TN, typename Tval>
inline static void
capped_mean_impl
    (int64_t d1, int64_t d2, int64_t d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    #ifdef CAPPED_MEAN_CUDA
    capped_mean_kernel<mode, TN, Tval><<<d1, d3>>>(d1, d2, d3, x, N, y);
    #else
    #ifdef _OPENMP
    // much better to parallelize over the batch dimension
    #pragma omp for schedule(static)
    #endif
    for (TN idx1=0; idx1 < d1; ++idx1)
        for (TN idx3=0; idx3 < d3; ++idx3)
            capped_mean_atom<mode, TN, Tval>(idx1, idx3, d1, d2, d3, x, N, y);
    #endif
}

// wrapper around the above for torch tensors
template<Mode mode, typename TN, typename Tval>
inline static void
call_capped_mean_impl
    (int64_t d1, int64_t d2, int64_t d3,
     const torch::Tensor &x,
     const torch::Tensor &N,
     torch::Tensor &y)
{
    // Tensor::element_size return bytes per element
    TORCH_CHECK(x.element_size() == sizeof(Tval), "type of x does not match Tval");
    TORCH_CHECK(N.element_size() == sizeof(TN), "type of N does not match TN");
    TORCH_CHECK(y.element_size() == sizeof(Tval), "type of y does not match Tval");

    // I believe torch::mean allows average over zero elements, giving NaN.
    // We are a bit more restrictive as we don't see a scenario in which this is
    // not a bug.
    // TODO

    const Tval *x_ptr = x.data_ptr<Tval>();
    const TN *N_ptr = N.data_ptr<TN>();
    Tval *y_ptr = y.data_ptr<Tval>();

    capped_mean_impl<mode, TN, Tval>(d1, d2, d3, x_ptr, N_ptr, y_ptr);
}

#ifdef CAPPED_MEAN_CUDA
#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#else
#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#endif

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_DEVICE(x); CHECK_CONTIGUOUS(x); } while(0)

// use this macro to dispatch capped_mean_impl according to types
// ... are the arguments passed to call_capped_mean_impl, which we omit to save some typing
#define DISPATCH_IMPL(mode, torch_TN, torch_Tval, ...)                                              \
    do {                                                                                            \
             if ((torch_TN==torch::ScalarType::Int) && (torch_Tval==torch::ScalarType::Float))      \
            call_capped_mean_impl<mode, int, float>(__VA_ARGS__);                                   \
        else if ((torch_TN==torch::ScalarType::Long) && (torch_Tval==torch::ScalarType::Float))     \
            call_capped_mean_impl<mode, int64_t, float>(__VA_ARGS__);                               \
        else if ((torch_TN==torch::ScalarType::Short) && (torch_Tval==torch::ScalarType::Float))    \
            call_capped_mean_impl<mode, int16_t, float>(__VA_ARGS__);                               \
        else if ((torch_TN==torch::ScalarType::Int) && (torch_Tval==torch::ScalarType::Double))     \
            call_capped_mean_impl<mode, int, double>(__VA_ARGS__);                                  \
        else if ((torch_TN==torch::ScalarType::Long) && (torch_Tval==torch::ScalarType::Double))    \
            call_capped_mean_impl<mode, int64_t, double>(__VA_ARGS__);                              \
        else if ((torch_TN==torch::ScalarType::Short) && (torch_Tval==torch::ScalarType::Double))   \
            call_capped_mean_impl<mode, int16_t, double>(__VA_ARGS__);                              \
        else /* fallthrough */                                                                      \
            TORCH_CHECK(false, "type combination not implemented");                                 \
    } while (0)

torch::Tensor
// need different names here to comply with the setup.py thing
#ifdef CAPPED_MEAN_CUDA
capped_mean_forward_CU
#else
capped_mean_forward
#endif
    (const torch::Tensor &x, const torch::Tensor &N, bool keepdim=false)
{
    CHECK_INPUT(x);
    CHECK_INPUT(N);

    // consistency
    check_shapes(x.sizes(), N.sizes());

    // figure out shape of the output
    int64_t out_dim;
    int64_t out_shape[8]; // whatever, this is gonna be enough
    get_out_shape(x.sizes(), N.sizes(), keepdim, out_dim, out_shape);

    int64_t d1, d2, d3;
    get_dims(x.sizes(), N.sizes(), d1, d2, d3);

    auto y = torch::empty(c10::IntArrayRef(out_shape, out_dim),
                          torch::TensorOptions()
                             .dtype(x.dtype())
                             .device(x.device())
                             .layout(torch::kStrided)
                             .requires_grad(false)
                         );

    CHECK_INPUT(y);

    // perform the computation
    DISPATCH_IMPL(FORWARD, N.scalar_type(), x.scalar_type(),
                  d1, d2, d3, x, N, y);

    return y;
}

torch::Tensor
#ifdef CAPPED_MEAN_CUDA
capped_mean_backward_CU
#else
capped_mean_backward
#endif
// Note that it is not convenient here to pass dim and keepdim, so we need to figure out
// what they are. But this is not complicated.
    (const torch::Tensor &x, const torch::Tensor &N, const torch::Tensor &grad)
{
    CHECK_INPUT(x);
    CHECK_INPUT(grad);
    CHECK_INPUT(N);

    // consistency
    check_shapes(x.sizes(), N.sizes(), grad.sizes());

    auto y = torch::empty_like(x, torch::TensorOptions().requires_grad(false));

    CHECK_INPUT(y);

    int64_t d1, d2, d3;
    get_dims(x.sizes(), N.sizes(), d1, d2, d3);
    
    TORCH_CHECK(x.scalar_type() == grad.scalar_type(),
                "x and grad expected to have same type");

    // perfom computation
    DISPATCH_IMPL(BACKWARD, N.scalar_type(), x.scalar_type(),
                  d1, d2, d3, grad, N, y);

    return y;
}

#undef CHECK_DEVICE
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
#undef DISPATCH_IMPL
