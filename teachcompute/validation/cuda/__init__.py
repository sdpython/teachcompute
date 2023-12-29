def cuda_version() -> str:
    """
    Returns the cuda version it was compiled with.
    If CUDA was not available, it retunrs `"0.0"`.
    """
    try:
        from .cuda_example_py import cuda_version as cv
    except ImportError:
        # No CUDA
        return "0.0"
    v = cv()
    return f"{v // 1000}.{(v % 1000) // 10}"
