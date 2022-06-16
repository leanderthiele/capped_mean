import torch

import capped_mean_cpu, capped_mean_cuda

class CappedMeanFunction(torch.autograd.Function) :
    
    @staticmethod
    def forward (ctx, x, N, keepdim=False) :
        ctx.save_for_backward(x, N)
        if x.is_cuda :
            return capped_mean_cuda.forward(x, N, keepdim)
        else :
            return capped_mean_cpu.forward(x, N, keepdim)

    @staticmethod
    def backward (ctx, grad) :
        x, N = ctx.saved_tensors
        if grad.is_cuda :
            xgrad = capped_mean_cuda.backward(x, N, grad)
        else :
            xgrad = capped_mean_cpu.backward(x, N, grad)
        
        # obviously, output is not differentiable w.r.t N and keepdim
        return xgrad, None, None


class CappedMean(torch.nn.Module) :
    
    def __init__(self) :
        super().__init__()

    def forward(self, x, N, keepdim=False) :
        return CappedMeanFunction.apply(x, N, False)
