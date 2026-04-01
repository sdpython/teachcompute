from pathlib import Path

from setuptools import Extension, setup

import nanobind

# Compiler flags for OpenMP
extra_compile_args = ["-O3", "-fopenmp", "-std=c++17", "-fvisibility=hidden"]
extra_link_args = ["-fopenmp"]

nb_include = nanobind.include_dir()
nb_src = Path(nanobind.source_dir()) / "nb_combined.cpp"
# nanobind bundles tsl/robin_map, needed when building nb_combined.cpp directly
nb_ext_include = str(
    Path(nanobind.include_dir()).parent / "ext" / "robin_map" / "include"
)

ext_modules = [
    Extension(
        "dotnanobind",
        sources=["dotnanobind.cpp", str(nb_src)],
        include_dirs=[nb_include, nb_ext_include],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="dotnanobind",
    version="0.1",
    author="Ton Nom",
    description="Nanobind + OpenMP dotnanobind",
    ext_modules=ext_modules,
    zip_safe=False,
)
