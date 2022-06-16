from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

setup(name='capped_mean',
      packages=['capped_mean'],
      ext_modules=[
                   CppExtension('capped_mean_cpu', ['capped_mean_cpu.cpp', ]),

                   # NOTE that there's a problem in which files can't be differentiated by extension only
                   CUDAExtension('capped_mean_cuda', ['capped_mean_cuda.cpp', 'capped_mean_cuda_CU.cu']),
                  ],
      cmdclass={'build_ext': BuildExtension}
     )
