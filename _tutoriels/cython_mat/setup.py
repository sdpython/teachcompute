from setuptools import setup, Extension
from Cython.Build import cythonize

ext = Extension(name="mmat", sources=["mmat.pyx", "mmat_impl.cpp"], language="c++")

setup(name="mmat", ext_modules=cythonize([ext], language_level="3"))
