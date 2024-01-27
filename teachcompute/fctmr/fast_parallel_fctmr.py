"""
Simple parallelization of *mapper* and *reducer* based on :epkg:`numba`.
:epkg:`Python` does not easily allow to parallelize functions
as the :epkg:`GIL` blocks most of the tentatives by imposing
a single tunnel for all allocations, creation of :epkg:`python`
objects. The language implements it but in practice it is not.
This file is just a tentative to use :epkg:`numba` to parallelize
a mapper but the number of round trip between :epkg:`python`
and compiled :epkg:`C` makes it difficult to write something
generic.
"""

from typing import Callable, Iterable, Optional
import numpy
from numba import jit, njit, prange


def create_array_numba(nb: int, sig: str) -> numpy.ndarray:
    """
    Creates an array of size nb knowing its signature.

    :param nb: integer
    :param signature: signature, ex: ``'f8'``
    :return: container
    """
    if sig == "f8":
        return numpy.empty(nb, dtype=numpy.float64)
    raise NotImplementedError("Cannot create a container for type '{0}'.".format(sig))


def fast_parallel_mapper(
    fct: Callable,
    gen: Iterable,
    chunk_size: int = 100000,
    parallel: bool = True,
    nogil: bool = False,
    nopython: bool = True,
    sigin: Optional[str] = None,
    sigout: Optional[str] = None,
) -> Iterable:
    """
    Parallelizes a mapper based on :epkg:`numba` and more specifically
    `Automatic parallelization with @jit <https://numba.pydata.org/
    numba-doc/dev/user/parallel.html>`_.
    This page indicates what :epkg:`numba` optimizes when
    it parallizes a map.

    :param fct: function
    :param gen: generator
    :param chunk_size: chunk size
    :param parallel: see
        `parallel
        <https://numba.pydata.org/numba-doc/latest/user/jit.html?highlight=nopython#parallel>`_
    :param nopython: see
        `nopython
        <https://numba.pydata.org/numba-doc/latest/user/jit.html?highlight=nopython#nopython>`_
    :param nogil: see
        `nogil
        <https://numba.pydata.org/numba-doc/latest/user/jit.html?highlight=nopython#nogil>`_
    :param sigin: signature of input type
    :param sigout: signature of output type
    :return: generator

    The parallelization can only happen if
    the array is known. So the function splits the
    array in chunck of size *chunk_size*.
    This tentative is not very efficient due to
    the genericity of the mapper. :epkg:`python`
    is not a good language to do that. See unit test
    `test_parallel_fctmr.py
    <https://github.com/sdpython/teachcompute/blob/main/_unittests/ut_fctmr/test_fast_parallel_fctmr.py>`_.
    """
    if sigin is not None and sigout is not None:
        sig1 = "{0}({1})".format(sigout, sigin)
        sig2 = "void(i8, {0}[:], {1}[:])".format(sigin, sigout)

        fct_jit = jit(
            sig1, nogil=nogil, parallel=parallel, nopython=nopython, cache=True
        )(fct)

        def loop(nb, inputs, outputs):
            "local function"
            for i in prange(nb):
                outputs[i] = fct_jit(inputs[i])

        loop_jit = njit(
            sig2, nogil=nogil, parallel=parallel, nopython=nopython, cache=True
        )(loop)
        inputs = create_array_numba(chunk_size, sigin)
        outputs = create_array_numba(chunk_size, sigout)

        done = 0
        for obs in gen:
            if done < len(inputs):
                inputs[done] = obs
                done += 1
            else:
                loop_jit(done, inputs, outputs)
                for out in outputs:
                    yield out
                done = 0
        if 0 < done < len(inputs):
            loop_jit(done, inputs, outputs)
            for out in outputs:
                yield out

    else:

        def loop(nb, inputs, outputs):
            "local function"
            for i in prange(nb):
                outputs[i] = fct_jit(inputs[i])

        loop_jit = None
        fct_jit = None

        inputs = None
        outputs = None

        done = 0
        for obs in gen:
            if inputs is None:
                inputs = [obs] * chunk_size
                outputs = [fct(obs)] * chunk_size
                loop_jit = njit(
                    nogil=nogil, parallel=parallel, nopython=nopython, cache=True
                )(loop)
                fct_jit = jit(
                    nogil=nogil, parallel=parallel, nopython=nopython, cache=True
                )(fct)

            if done < len(inputs):
                inputs[done] = obs
                done += 1
            else:
                loop_jit(done, inputs, outputs)
                for out in outputs:
                    yield out
                done = 0
        if 0 < done < len(inputs):
            loop_jit(done, inputs, outputs)
            for out in outputs:
                yield out
