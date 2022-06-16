/* This header implements the core functionality.
 *
 * Define CAPPED_MEAN_CUDA before including to get the CUDA version.
 */

// TODO maybe we can template on bool forward/backward for more compact code

#include <torch/extension.h>

#include "atom.h"
#include "shapes.h"

#ifdef CAPPED_MEAN_CUDA
template<typename TN, typename Tval>
__global__
static void
capped_mean_forward_kernel
    (TN d1, TN d2, TN d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    TN idx1 = blockIdx.x,
       idx3 = threadIdx.x;

    capped_mean_forward_atom<TN, Tval>(idx1, idx3,
                                       d1, d2, d3,
                                       x, N, y);
}

template<typename TN, typename Tval>
__global__
static void
capped_mean_backward_kernel
    (TN d1, TN d2, TN d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    TN idx1 = blockIdx.x,
       idx3 = threadIdx.x;

    capped_mean_backward_atom<TN, Tval>(idx1, idx3,
                                        d1, d2, d3,
                                        x, N, y);
}

// need to instantiate for all possible types
#define INSTANTIATE_KERNELS(TN, Tval)              \
    template __global__ void                       \
    capped_mean_forward_kernel <TN, Tval>          \
    (TN, TN, TN, const Tval *, const TN *, Tval *);\
    template __global__ void                       \
    capped_mean_backward_kernel <TN, Tval>         \
    (TN, TN, TN, const Tval *, const TN *, Tval *)

// at the moment, we don't do this for that many types, can always
// add if something comes up
INSTANTIATE_KERNELS(int, float);
INSTANTIATE_KERNELS(int64_t, float);

#undef INSTANTIATE_KERNEL

#endif // CAPPED_MEAN_CUDA

template<typename TN, typename Tval>
static void
capped_mean_forward_impl
    (TN d1, TN d2, TN d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    #ifdef CAPPED_MEAN_CUDA
    capped_mean_forward_kernel<TN, Tval><<<d1, d3>>>(d1, d2, d3, x, N, y);
    #else
    for (TN idx1=0; idx1 != d1; ++idx1)
        for (TN idx3=0; idx3 != d3; ++idx3)
            capped_mean_forward_atom(idx1, idx3, d1, d2, d3, x, N, y);
    #endif
}

template<typename TN, typename Tval>
static void
capped_mean_backward_impl
    (TN d1, TN d2, TN d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    #ifdef CAPPED_MEAN_CUDA
    capped_mean_backward_kernel<TN, Tval><<<d1, d3>>>(d1, d2, d3, x, N, y);
    #else
    for (TN idx1=0; idx1 != d1; ++idx1)
        for (TN idx3=0; idx3 != d3; ++idx3)
            capped_mean_backward_atom(idx1, idx3, d1, d2, d3, x, N, y);
    #endif
}

#ifdef CAPPED_MEAN_CUDA
#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#else
#define CHECK_DEVICE(x) TORCH_CHECK(x.device().is_cpu(), #x " must be a CPU tensor")
#endif

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) do { CHECK_DEVICE(x); CHECK_CONTIGUOUS(x); } while(0)

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
                             .requires_grad(false) // TODO
                         );

    CHECK_INPUT(y);

    // TODO apparently, in the following .type() is deprecated

    // perform the computation
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "types other than float not implemented yet");
    
    #define CALL_WITH_TYPES(TN, Tval)                          \
        capped_mean_forward_impl<TN, Tval>(d1, d2, d3,         \
                                           x.data_ptr<Tval>(), \
                                           N.data_ptr<TN>(),   \
                                           y.data_ptr<Tval>())

    // NOTE I have checked with the header that
    // Int = int, Long = int64_t
    if (N.scalar_type() == torch::ScalarType::Int)
        CALL_WITH_TYPES(int, float);
    else if (N.scalar_type() == torch::ScalarType::Long)
        CALL_WITH_TYPES(int64_t, float);
    else
        TORCH_CHECK(false, "N has unsupported type");

    #undef CALL_WITH_TYPES

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

    auto y = torch::empty_like(x);

    CHECK_INPUT(y);

    int64_t d1, d2, d3;
    get_dims(x.sizes(), N.sizes(), d1, d2, d3);
    
    TORCH_CHECK(x.scalar_type() == grad.scalar_type(),
                "x and grad expected to have same type");
    TORCH_CHECK(x.scalar_type() == torch::ScalarType::Float,
                "types other than float not implemented yet");

    #define CALL_WITH_TYPES(TN, Tval)                              \
        capped_mean_backward_impl<TN, Tval>(d1, d2, d3,            \
                                            grad.data_ptr<Tval>(), \
                                            N.data_ptr<TN>(),      \
                                            y.data_ptr<Tval>())

    if (N.scalar_type() == torch::ScalarType::Int)
        CALL_WITH_TYPES(int, float);
    else if (N.scalar_type() == torch::ScalarType::Long)
        CALL_WITH_TYPES(int64_t, float);
    else
        TORCH_CHECK(false, "N has unsupported type");

    #undef CALL_WITH_TYPES

    return y;
}

#undef CHECK_DEVICE
#undef CHECK_CONTIGUOUS
#undef CHECK_INPUT
