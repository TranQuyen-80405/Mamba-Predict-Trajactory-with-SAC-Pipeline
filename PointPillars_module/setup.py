import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME


def _build_extension():
    use_cuda = bool((CUDA_HOME and os.path.isdir(CUDA_HOME)) and (torch.version.cuda is not None))
    if os.getenv("FORCE_CUDA", "0") == "1":
        use_cuda = True

    common_sources = [
        "pointpillars/ops/voxelization/voxelization.cpp",
        "pointpillars/ops/voxelization/voxelization_cpu.cpp",
    ]
    if use_cuda:
        return CUDAExtension(
            name="pointpillars.ops.voxel_op",
            sources=common_sources + ["pointpillars/ops/voxelization/voxelization_cuda.cu"],
            define_macros=[("WITH_CUDA", None)],
            extra_compile_args={
                "cxx": [],
                "nvcc": ["-allow-unsupported-compiler"],
            },
        )

    return CppExtension(
        name="pointpillars.ops.voxel_op",
        sources=common_sources,
    )


setup(
    name="pointpillars",
    version="0.1",
    packages=find_packages(),
    ext_modules=[_build_extension()],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
