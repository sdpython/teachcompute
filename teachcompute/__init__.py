# -*- coding: utf-8 -*-

__version__ = "0.2.0"
__author__ = "Xavier Dupré"
__github__ = "https://github.com/sdpython/teachcompute"
__url__ = "https://sdpython.github.io/doc/teachcompute/dev/"
__license__ = "MIT License"


def check_installation(val: bool = False, verbose: bool = False):
    """
    Quickly checks the installation works.

    :param val: checks that a couple of functions
        in submodule validation are working
    :param verbose: prints out which verifications is being processed
    """
    import datetime

    def local_print(msg):
        t = datetime.datetime.now().time()
        print(msg.replace("[check_installation]", f"[check_installation] {t}"))

    if verbose:
        local_print("[check_installation] --begin")
    assert isinstance(get_cxx_flags(), str)

    if val:
        if verbose:
            local_print("[check_installation] --val")
            local_print("[check_installation] import numpy")
        import numpy

        if verbose:
            local_print("[check_installation] import teachcompute")
        from teachcompute.validation.cython.vector_function_cy import vector_sum_cy

        a = (
            ((numpy.arange(9).astype(numpy.float32) - 5))
            .astype(numpy.float32)
            .reshape((3, -1))
        )
        c = vector_sum_cy(a)
        assert isinstance(c, float)
        assert c == -9
        if verbose:
            local_print("[check_installation] cast_float32_to_e4m3fn")
        if verbose:
            local_print("[check_installation] --done")


def has_cuda() -> bool:
    """
    Tells if cuda is available.
    """
    from ._config import HAS_CUDA

    return HAS_CUDA == 1


def cuda_version() -> str:
    """
    Tells which version of CUDA was used to build the CUDA extensions.
    """
    assert has_cuda(), "CUDA extensions are not available."
    from ._config import CUDA_VERSION

    return CUDA_VERSION


def cuda_version_int() -> tuple:
    """
    Tells which version of CUDA was used to build the CUDA extensions.
    It returns `(0, 0)` if CUDA is not present.
    """
    if not has_cuda():
        return (0, 0)
    from ._config import CUDA_VERSION

    if not isinstance(CUDA_VERSION, str):
        return tuple()

    spl = CUDA_VERSION.split(".")
    return tuple(map(int, spl))


def compiled_with_cuda() -> bool:
    """
    Checks it was compiled with CUDA.
    """
    try:
        from .validation.cuda import cuda_example_py

        return cuda_example_py is not None
    except ImportError:
        return False


def get_cxx_flags() -> str:
    """
    Returns `CXX_FLAGS`.
    """
    from ._config import CXX_FLAGS

    return CXX_FLAGS


def get_stdcpp() -> int:
    """
    Returns `CMAKE_CXX_STANDARD`.
    """
    from ._config import CMAKE_CXX_STANDARD

    return CMAKE_CXX_STANDARD
