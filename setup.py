from setuptools import Extension, setup

from build_ext import BuildExtensions

ext_modules = [
    Extension(
        "backend.mps",
        sources=["include/mps.mm", "include/wrapper.cpp"],
        include_dirs=[],
        extra_compile_args=["-std=c++17", "-fobjc-arc", "-ObjC++"],
        extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
    ),
    Extension(
        "backend.tensor",
        sources=["include/tensor.mm"],
        include_dirs=[],
        extra_compile_args=["-std=c++17", "-fobjc-arc", "-ObjC++"],
        extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
    ),
]
setup(
    cmdclass={
        "build_ext": BuildExtensions,
    },
    ext_modules=ext_modules,
)
