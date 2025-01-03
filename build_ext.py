import os
import shutil
import subprocess
from distutils.unixccompiler import UnixCCompiler

from setuptools.command.build_ext import build_ext

# Allow .mm files to be recognized as Objective-C++.
UnixCCompiler.src_extensions.append(".mm")
UnixCCompiler.language_map[".mm"] = "objc++"

METAL_SOURCES = ["src/Shaders.metal"]
METAL_LIB_NAME = "shader.metallib"


class BuildExtensions(build_ext):
    def run(self):
        build_dir = os.path.join(os.path.abspath(self.build_lib), "backend")
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)
        init_path = os.path.join(build_dir, "__init__.py")
        if not os.path.exists(init_path):
            open(init_path, "a").close()

        self.compile_metal(METAL_SOURCES, METAL_LIB_NAME, build_dir)
        super().run()
        to = os.path.join(os.getcwd(), "backend")
        if os.path.exists(to):
            shutil.rmtree(to)
        print(f"copying {build_dir} -> {to}")
        shutil.move(build_dir, to)

    def compile_metal(self, metal_sources, metallib_name, build_dir):
        """Shaders.metal -> shader.air -> shader.metallib"""
        metal_file = metal_sources[0]
        air_file = metal_file.replace(".metal", ".air")
        metallib_file = os.path.join("src", metallib_name)
        print("Compiling Metal shader to .air...")
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metal", "-c", metal_file, "-o", air_file]
        )
        print("Compiling .air to .metallib...")
        subprocess.check_call(
            ["xcrun", "-sdk", "macosx", "metallib", air_file, "-o", metallib_file]
        )
        if os.path.exists(air_file):
            os.remove(air_file)
        target_metallib = os.path.join(build_dir, metallib_name)
        self.copy_file(metallib_file, target_metallib)
        if os.path.exists(metallib_file):
            os.remove(metallib_file)
