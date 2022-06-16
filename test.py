import time
import torch
from capped_mean import capped_mean
import warnings

# tuples shape, dim
# If the first two dimensions match we employ an artificial transpose
# to test how expensive the calls to .contiguous() are
TEST_CASES = [( (256, 128, 64), 1 ),
              ( (32, 32, 128, 4, 5), 2),
              ( (256, 128, 2), 1 ),
              ( (128, 128, 64), 1 ),
             ]

def run (nruns, f, shape, dim, device) :
    """ the testing function
    nruns ... number of times to run for timing
    f ... callable, with the arguments
               x ... torch float tensor of shape [shape]
               N ... torch int tensor of shape [shape][:dim]
          Implements the capped mean
    shape ... the shape of x
    dim ... along which direction to collapse
    device ... where to do the computation

    Output is reproducible so can be compared.
    """

    rng = torch.Generator(device=device).manual_seed(137)
    x = torch.rand(*shape, requires_grad=True,
                   dtype=torch.float32, generator=rng, device=device)
    N = torch.randint(low=1, high=shape[dim], size=shape[:dim],
                      dtype=torch.int64, generator=rng, device=device)

    # when the first two dimensions are equal, we transpose to test
    # cost of making things contiguous
    # have confirmed that CappedMean has to call .contiguous every one of the nruns
    transf = lambda y: y if y.shape[0] != y.shape[1] else y.transpose(0, 1)

    start = time.time()
    for _ in range(nruns) :
        f(transf(x), N)
    forward_time = ( time.time() - start ) / nruns

    start = time.time()
    for _ in range(nruns) :
        # need to collapse to scalar here, but overhead from sum should be small
        torch.sum(f(x, N)).backward()
    forward_backward_time = ( time.time() - start ) / nruns

    # initialize again so we don't accumulate small errors in the gradients
    x = torch.rand(*shape, requires_grad=True,
                   dtype=torch.float32, generator=rng, device=device)
    y = f(transf(x), N)
    
    # we put a non-linear function here in addition for more robust test
    torch.sum(torch.sin(y)).backward()
    g = x.grad

    return forward_time * 1e6, forward_backward_time * 1e6, y, g

if __name__ == '__main__' :

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on: {device}')

    my_f = capped_mean
    torch_f = lambda x, N : torch.reshape(torch.stack([torch.mean(x_[:N_, ...], dim=0) 
                                                      for x_, N_ in
                                                      zip(x.reshape(-1, *x.shape[N.dim():]), N.flatten())]),
                                          (*x.shape[:N.dim()], *x.shape[N.dim()+1:]))
    wrong_f = lambda x, N : torch.mean(x, dim=N.dim())

    for shape, dim in TEST_CASES :

        print(f'\nshape={shape}, dim={dim}')

        if shape[0] == shape[1] :
            print('*** Doing artificial transpose to test cost of contiguous.')

        # additional check using torch.autograd.gradcheck
        print('\tRunning torch.autograd.gradcheck...')
        rng = torch.Generator(device=device).manual_seed(42)
        x = torch.rand(*shape, requires_grad=True,
                       dtype=torch.float32, generator=rng, device=device)
        N = torch.randint(low=1, high=shape[dim], size=shape[:dim],
                          dtype=torch.int64, generator=rng, device=device)
        f = lambda x_, N_=N : my_f(x_, N_)
        # this is all linear so the precision actually doesn't matter
        warnings.filterwarnings('ignore', category=UserWarning,
                                message='Input #0 requires gradient and is not a '\
                                        'double precision floating point or complex. '\
                                        'This check will likely fail if all the inputs '\
                                        'are not of double precision floating point or complex.')
        torch.autograd.gradcheck(f, x, raise_exception=True, fast_mode=True)
        print('\t... passed torch.autograd.gradcheck')

        my_ft, my_fbt, my_y, my_g = run(1000, my_f, shape, dim, device)
        torch_ft, torch_fbt, torch_y, torch_g = run(100, torch_f, shape, dim, device)
        wrong_ft, wrong_fbt, _, _ = run(1000, wrong_f, shape, dim, device)

        speedup_f = torch_ft / my_ft
        speedup_fb = torch_fbt / my_fbt

        correct_forward = 'CORRECT' if torch.allclose(my_y, torch_y) else 'WRONG'
        correct_backward = 'CORRECT' if torch.allclose(my_g, torch_g) else 'WRONG'

        print(f'\tforward: {correct_forward}  backward: {correct_backward}')
        print(f'\tforward timing [us]: mine={my_ft:.2f} interpreted={torch_ft:.2f} native(wrong)={wrong_ft:.2f} '\
              f'--> speedup: x{speedup_f:.0f}')
        print(f'\tforward+backward timing [us]: mine={my_fbt:.2f} interpreted={torch_fbt:.2f} native(wrong)={wrong_fbt:.2f} '\
              f'--> speedup: x{speedup_fb:.0f}')
