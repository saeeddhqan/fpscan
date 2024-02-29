
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess

setup(
    name='something_weird',
    ext_modules=[
        CUDAExtension('something_weird', [
            'main.cpp',
            'dimwise/dimwise_scan_final.cu',
            'full_scan/full_scan.cu',
        ],
        extra_compile_args={'cxx': ['-O3'],
                             'nvcc': ['-O3', '-lineinfo', '--use_fast_math', '-std=c++17', '--ptxas-options=-v',
                             '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF_OPERATORS__',
                             '-U__CUDA_NO_BFLOAT16_OPERATORS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                             '--threads', '2']
        })
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    version='0.0.0',
    description='Contraction parallel scan',
    url='https://github.com/saeeddhqan/fpscan',
    author='Saeed',
    license='Apache 2.0',
)
