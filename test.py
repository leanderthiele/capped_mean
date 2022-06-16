import time
import torch
from capped_mean import CappedMean

# tuples shape, dim
TEST_CASES = [( (256, 128, 64), 1 ),
              ( (32, 32, 128, 4, 5), 2),
              ( (256, 128, 1), 1 ),
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

    rng = torch.manual_seed(137)
    x = torch.rand(*shape, requires_grad=True,
                   dtype=torch.float32, generator=rng, device=device)
    N = torch.randint(low=1, high=shape[dim], size=shape[:dim],
                      dtype=torch.int64, generator=rng, device=device)

    start = time.time()
    for _ in range(nruns) :
        f(x, N)
    forward_time = ( time.time() - start ) / nruns

    start = time.time()
    for _ in range(nruns) :
        # need to collapse to scalar here, but overhead from sum should be small
        torch.sum(f(x, N)).backward()
    forward_backward_time = ( time.time() - start ) / nruns

    # initialize again so we don't accumulate small errors in the gradients
    x = torch.rand(*shape, requires_grad=True,
                   dtype=torch.float32, generator=rng, device=device)
    y = f(x, N)
    torch.sum(y).backward()
    g = x.grad

    return forward_time * 1e6, forward_backward_time * 1e6, y, g

if __name__ == '__main__' :

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    my_f = CappedMean()
    torch_f = lambda x, N : torch.reshape(torch.stack([torch.mean(x_[:N[ii], ...], dim=0) 
                                                      for ii, x_ in enumerate(x.reshape(-1, *x.shape[N.dim():]))),
                                          (*x.shape[:N.dim()], *x.shape[N.dim()+1:]))
    wrong_f = lambda x, N : torch.mean(x, dim=N.dim())

    print(f'On device: {device}')

    for shape, dim in TEST_CASES :

        print(f'shape={shape}, dim={dim}')

        my_ft, my_fbt, my_y, my_g = run(1000, my_f, shape, dim, device)
        torch_ft, torch_fbt, torch_y, torch_g = run(1000, torch_f, shape, dim, device)
        wrong_ft, wrong_ftb, _, _ = run(1000, wrong_f, shape, dim, device)

        correct_forward = 'CORRECT' if torch.allclose(my_y, torch_y) else 'WRONG'
        correct_backward = 'CORRECT' if torch.allclose(my_g, torch_g) else 'WRONG'

        print(f'\tforward: {correct_forward}  backward: {correct_backward}')
        print(f'\tforward timing [us]: mine={my_ft:%.2f} interpreted={torch_ft:%.2f} native(wrong)={wrong_ft:%.2f}')
        print(f'\tforward+backward timing [us]: mine={my_ftb:%.2f} interpreted={torch_ftb:%.2f} native(wrong)={wrong_ftb:%.2f}')
