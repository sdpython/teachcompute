from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="dotcy",
    sources=["dotcy.pyx"],
    include_dirs=[numpy.get_include()],
)

setup(
    name="dotcy",
    ext_modules=cythonize([ext], language_level="3"),
)
