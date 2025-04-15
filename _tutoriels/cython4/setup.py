from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="dotcy",
    sources=["dotcy.pyx"],
    include_dirs=[numpy.get_include()],
    language="c++",
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="dotcy",
    ext_modules=cythonize([ext], language_level="3"),
)
