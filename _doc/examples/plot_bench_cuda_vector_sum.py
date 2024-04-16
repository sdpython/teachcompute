"""
.. _l-example-cuda-vector-sum:

Measuring CUDA performance with a vector sum
============================================

The objective is to measure the summation of all elements from a tensor.

::

    nsys profile python _doc/examples/plot_bench_cuda_vector_sum.py

Vector Add
++++++++++
"""

from tqdm import tqdm
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame
from teachcompute.ext_test_case import measure_time, unit_test_going
import torch

has_cuda = torch.cuda.is_available()

try:
    from teachcompute.validation.cuda.cuda_example_py import vector_sum0, vector_sum_atomic, vector_sum6
except ImportError:
    has_cuda = False


def wrap_cuda_call(f, values):
    torch.cuda.nvtx.range_push(f"CUDA f={f.__name__} dim={values.size}")
    res = f(values)
    torch.cuda.nvtx.range_pop()
    return res


obs = []
dims = [2**10, 2**15, 2**20, 2**25, 2**28]
if unit_test_going():
    dims = [10, 20, 30]
for dim in tqdm(dims):
    values = numpy.ones((dim,), dtype=numpy.float32).ravel()

    if has_cuda:
        for f in [vector_sum0, vector_sum_atomic, vector_sum6]:
            if f == vector_sum_atomic and dim > 2**20:
                continue
            diff = numpy.abs(wrap_cuda_call(f, values) - (values.sum()))
            res = measure_time(lambda: wrap_cuda_call(f, values), max_time=0.5)

            obs.append(
                dict(
                    dim=dim,
                    size=values.size,
                    time=res["average"],
                    fct=f"CUDA-{f.__name__}",
                    time_per_element=res["average"] / dim**2,
                    diff=diff,
                )
            )

    diff = 0
    res = measure_time(lambda: values.sum(), max_time=0.5)

    obs.append(
        dict(
            dim=dim,
            size=values.size,
            time=res["average"],
            fct="numpy",
            time_per_element=res["average"] / dim**2,
            diff=0,
        )
    )


df = DataFrame(obs)
piv = df.pivot(index="dim", columns="fct", values="time_per_element")
print(piv)


##############################################
# Plots
# +++++

piv_diff = df.pivot(index="dim", columns="fct", values="diff")
piv_time = df.pivot(index="dim", columns="fct", values="time")

fig, ax = plt.subplots(1, 3, figsize=(12, 6))
piv.plot(ax=ax[0], logx=True, title="Comparison between two summation")
piv_diff.plot(ax=ax[1], logx=True, logy=True, title="Summation errors")
piv_time.plot(ax=ax[2], logx=True, logy=True, title="Total time")
fig.tight_layout()
fig.savefig("plot_bench_cuda_vector_sum.png")

##############################################
# CUDA seems very slow but in fact, all the time is spent
# in moving the data from the CPU memory (Host) to the GPU memory (device).
#
# .. image:: ../images/nsight_vector_sum.png
#
