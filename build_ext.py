import os
import shutil
import subprocess
from distutils.unixccompiler import UnixCCompiler

from setuptools.command.build_ext import build_ext

# Allow .mm files to be recognized as Objective-C++.
UnixCCompiler.src_extensions.append(".mm")
UnixCCompiler.language_map[".mm"] = "objc++"

METAL_SOURCE = "src/kernels"
METAL_LIB_NAME = "kernels.metallib"


class BuildExtensions(build_ext):
    def run(self):
        build_dir = os.path.join(os.path.abspath(self.build_lib), "backend")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        init_path = os.path.join(build_dir, "__init__.py")
        if not os.path.exists(init_path):
            open(init_path, "a").close()

        self.compile_metal(
            [path for path in os.listdir(METAL_SOURCE) if path.endswith(".metal")],
            METAL_LIB_NAME,
            build_dir,
        )
        super().run()
        to = os.path.join(os.getcwd(), "backend")
        if os.path.exists(to):
            shutil.rmtree(to)
        print(f"copying {build_dir} -> {to}")
        shutil.move(build_dir, to)

    def compile_metal(self, metal_sources, metallib_name, build_dir):
        """Shaders.metal -> shader.air -> shader.metallib"""

        if not os.path.exists("build/"):
            os.mkdir("build/")
        air_sources = [
            os.path.join("build", file.replace(".metal", ".air"))
            for file in metal_sources
        ]
        metallib_file = os.path.join("src", metallib_name)
        print("Compiling Metal shader to .air...")
        for file in metal_sources:
            subprocess.check_call(
                [
                    "xcrun",
                    "-sdk",
                    "macosx",
                    "metal",
                    "-c",
                    os.path.join(METAL_SOURCE, file),
                    "-o",
                    os.path.join("build", file.replace(".metal", ".air")),
                ]
            )

        print("Compiling .air to .metallib...")
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metallib", *air_sources, "-o", metallib_file]
        )
        for air_file in air_sources:
            if os.path.exists(air_file):
                os.remove(air_file)
        target_metallib = os.path.join(build_dir, metallib_name)
        self.copy_file(metallib_file, target_metallib)
        if os.path.exists(metallib_file):
            os.remove(metallib_file)
