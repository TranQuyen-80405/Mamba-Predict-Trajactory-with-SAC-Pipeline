from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointpillars.ops.voxel_op',
            sources=[
                'pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)],
            # -allow-unsupported-compiler lets nvcc build against MSVC
            # toolsets newer than the CUDA version officially supports
            # (e.g. MSVC 14.5x + CUDA 12.8). Required for the voxelization
            # kernel to compile on Windows with VS 2026 preview toolchain.
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-allow-unsupported-compiler'],
            },
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)
