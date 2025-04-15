from setuptools import setup, Extension
import sys
import os
import pybind11

from pybind11.setup_helpers import Pybind11Extension, build_ext

# Compiler flags for OpenMP
extra_compile_args = ['-O3', '-fopenmp']
extra_link_args = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "dot11",
        ["dot11.cpp"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        cxx_std=17,
    ),
]

setup(
    name="dot11",
    version="0.1",
    author="Ton Nom",
    description="Pybind11 + OpenMP dot11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
