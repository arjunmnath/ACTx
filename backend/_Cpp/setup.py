import os
from distutils.unixccompiler import UnixCCompiler

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Register .mm extension with UnixCCompiler
UnixCCompiler.src_extensions.append(".mm")
UnixCCompiler.language_map[".mm"] = "objc++"

# clang++ -std=c++17 -ObjC++ -o out mps.mm wrapper.cpp -framework Metal -framework Foundation


class CustomBuildExt(build_ext):
    def build_extensions(self):
        if not os.path.exists("backend"):
            os.makedirs("backend")
        init_path = os.path.join("backend", "__init__.py")
        if not os.path.exists(init_path):
            open(init_path, "a").close()

        super().build_extensions()


setup(
    name="backend",
    ext_modules=[
        Extension(
            "backend.mps",
            sources=["src/mps.mm", "src/wrapper.cpp"],
            include_dirs=[],  # Add any necessary include directories
            extra_compile_args=[
                "-std=c++17",
                "-fobjc-arc",
                "-ObjC++",
            ],  # Additional compilation flags if needed
            extra_link_args=["-framework", "Metal", "-framework", "Foundation"],
        )
    ],
    package_data={
        "backend": ["src/Shaders.metal"],  # Include the file in the backend package
    },
    include_package_data=True,  # Ensure package_data is included
)
