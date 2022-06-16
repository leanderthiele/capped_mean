import torch
from torch import Tensor
from torch.types import _bool

import capped_mean_cpu, capped_mean_cuda

# NOTE
# One may be worried about the cost of the contiguous calls, which are necessary
# because the compiled library only operates on raw memory.
# However, I have implemented a check for this in test.py and the cost is marginal.

class CappedMeanFunction(torch.autograd.Function) :

    @staticmethod
    def _dispatcher (function_name, x, *args) :
        # small helper to abstract away the device
        m = capped_mean_cuda if x.is_cuda else capped_mean_cpu
        f = getattr(m, function_name)
        return f(x, *args)
    
    @staticmethod
    def forward (ctx, x, N, keepdim) :
        x = x.contiguous()
        N = N.contiguous()
        ctx.save_for_backward(x, N)
        return CappedMeanFunction._dispatcher('forward', x, N, keepdim)

    @staticmethod
    def backward (ctx, grad) :
        x, N = ctx.saved_tensors # these are contiguous by construction
        grad = grad.contiguous()
        xgrad = CappedMeanFunction._dispatcher('backward', x, N, grad)
        
        # obviously, output is not differentiable w.r.t. N and keepdim
        return xgrad, None, None


class CappedMean(torch.nn.Module) :
    
    def __init__(self) :
        super().__init__()

    def forward(self, x: Tensor, N: Tensor, keepdim: _bool=False) -> Tensor :
        return CappedMeanFunction.apply(x, N, False)
