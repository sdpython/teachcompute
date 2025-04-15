from libcpp.vector cimport vector
cimport cython

cdef extern from "c_primes.h":
    vector[int] c_primes(int n)


def primes(int nb_primes):
    return c_primes(nb_primes)
