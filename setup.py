
import os
import sys

from setuptools import setup

BUILD_COMMANDS = {
    "build",
    "build_ext",
    "bdist_wheel",
    "bdist_egg",
    "develop",
    "install",
    "editable_wheel",
}
_wants_build = BUILD_COMMANDS.intersection(sys.argv)

ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    CSRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    ext_modules = [
        CUDAExtension(
            name="fpscan._C",
            sources=[
                os.path.join("csrc", "main.cpp"),
                os.path.join("csrc", "dimwise", "dimwise_scan_final.cu"),
                os.path.join("csrc", "full_scan", "full_scan.cu"),
            ],
            include_dirs=[CSRC],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "--use_fast_math",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--threads",
                    "2",
                ],
            },
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
except ImportError as exc:
    if _wants_build:
        raise SystemExit(
            "Building fpscan requires PyTorch to be installed first and visible "
            "to the build. Install torch (matching your CUDA), then run:\n"
            "    pip install . --no-build-isolation\n"
            f"(import error: {exc})"
        )

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
