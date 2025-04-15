from setuptools import setup
from Cython.Build import cythonize

ext = Extension(
    name="primes",
    sources=["primes.pyx", "c_primes.cpp"],
    language="c++"
)

setup(
    name="primes",
    ext_modules=cythonize([ext], language_level="3")
)
