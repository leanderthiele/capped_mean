/* Contains element-wise operations, to put into CUDA kernels
 * or CPU for loops
 */

// TODO maybe we can template on bool forward/backward for more compact code

#include <type_traits>

// use as template parameters
enum Mode { FORWARD, BACKWARD };

/* We compute the following:
 * 
 * Given tensor x [ d1, d2, d3 ] (floating point),
 * and tensor N [ d1 ],
 * output tensor y [ d1, d3 ] where
 * y[i, j] = mean(x[i, :N[i], j]
 *
 * This header contains the core function to compute y[i, j].
 *
 * It is adapted in such a way that it works well in CUDA kernels.
 * Define the macro CAPPED_MEAN_CUDA before including this header
 * to have the CUDA version.
 *
 * For the backward function, the shapes are essentially reversed.
 *
 * The input x is now the gradient we are getting passed, of shape [ d1, d3 ],
 * while the output y is the array we are writing into, of shape [ d1, d2, d3 ]
 *
 * The derivative itself is obviously pretty simple.
 */

template<Mode mode, typename TN, typename Tval>
#ifdef CAPPED_MEAN_CUDA
__device__ __forceinline__
#else
inline
#endif
static void
capped_mean_atom
    (TN idx1, TN idx3,
     TN d1, TN d2, TN d3,
     const Tval * __restrict__ x,
     const TN * __restrict__ N,
     Tval * __restrict__ y)
{
    static_assert(std::is_integral<TN>::value);
    static_assert(std::is_floating_point<Tval>::value);

    #ifdef CAPPED_MEAN_CUDA
    if (idx1 >= d1 || idx3 >= d3)
        return;
    #endif

    TN this_N = N[idx1],
       idxout = idx1 * d3 + idx3;

    if constexpr (mode == FORWARD)
    {
        #ifdef CAPPED_MEAN_CUDA
        TN maxidx2 = d2;
        #else
        TN maxidx2 = this_N;
        #endif

        y[idxout] = 0.0;

        for (TN idx2=0; idx2 != maxidx2; ++idx2)
            y[idxout] += x[idx1*d2*d3 + idx2*d3 + idx3]
                         #ifdef CAPPED_MEAN_CUDA
                         * ( (idx2 < this_N) ? 1.0 : 0.0 )
                         #endif
                         ;

        y[idxout] /= (Tval)this_N;
    }
    else if constexpr (mode == BACKWARD)
    {
        for (TN idx2=0; idx2 != d2; ++idx2)
            y[idx1*d2*d3 + idx2*d3 + idx3] = (idx2<this_N) ? 
                                             x[idxout] / (Tval)this_N :
                                             0.0;
    }
}
