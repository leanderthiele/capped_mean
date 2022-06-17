/* One useful function to figure out the various tensor shapes */

#include <torch/extension.h>

inline static void
check_shapes
    (const c10::IntArrayRef x_shape, const c10::IntArrayRef N_shape)
// consistency checks for forward pass
{
    size_t dim = N_shape.size();

    TORCH_CHECK(dim>0, "dim(N) == 0 not sure if supported.");
    TORCH_CHECK(dim<=x_shape.size(), "Cannot have dim(N)>dim(x)");

    for (size_t ii=0; ii < dim; ++ii)
        TORCH_CHECK(x_shape[ii]==N_shape[ii], "x and N shapes do not match");
}

inline static void
check_shapes
    (const c10::IntArrayRef x_shape, const c10::IntArrayRef N_shape, const c10::IntArrayRef grad_shape)
// consistency checks for backward pass
{
    check_shapes(x_shape, N_shape);

    size_t dim = N_shape.size();

    for (size_t ii=0; ii < dim; ++ii)
        TORCH_CHECK(x_shape[ii]==grad_shape[ii],
                    "unexpectedly, x and grad have differing shapes before dim");

    bool keepdim = x_shape.size() == grad_shape.size();
    for (size_t ii=dim+1; ii < x_shape.size(); ++ii)
        TORCH_CHECK(x_shape[ii]==grad_shape[ii-((keepdim) ? 0 : 1)],
                    "unexpectedly, x and grad have different shapes after dim");
}

inline static void
get_out_shape
    (const c10::IntArrayRef x_shape, const c10::IntArrayRef N_shape,
     bool keepdim,
     int64_t &out_dim, int64_t *out_shape)
// figures out shape of the output
{
    // this is the dimension we are averaging over
    size_t dim = N_shape.size();

    out_dim = (keepdim) ? x_shape.size() : x_shape.size()-1;
    for (size_t ii=0; ii != dim; ++ii)
        out_shape[ii] = x_shape[ii];

    if (keepdim)
    {
        out_shape[dim] = 1;
        for (size_t ii=dim+1; ii < x_shape.size(); ++ii)
            out_shape[ii] = x_shape[ii];
    }
    else
    {
        for (size_t ii=dim+1; ii < x_shape.size(); ++ii)
            out_shape[ii-1] = x_shape[ii];
    }
}

inline static void
get_dims
    (const c10::IntArrayRef x_shape, const c10::IntArrayRef N_shape,
     idx_t &d1, idx_t &d2, idx_t &d3)
// figures out our canonical shape
// the input can be reshaped into [ d1, d2, d3 ], where d2 is the same dimension
// as the one we want to take the mean over (i.e. dim)
{
    size_t dim = N_shape.size();

    d2 = x_shape[dim];

    d1 = d3 = 1;

    for (size_t ii=0; ii < dim; ++ii)
        d1 *= x_shape[ii];

    for (size_t ii=dim+1; ii < x_shape.size(); ++ii)
        d3 *= x_shape[ii];
}
