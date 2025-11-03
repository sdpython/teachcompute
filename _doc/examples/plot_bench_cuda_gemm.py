"""
.. _l-example-cuda-gemm:

Comparing GEMM implementation
=============================

It is not exactly GEMM but MatMul with transpose attributes.

::

    nsys profile python _doc/examples/plot_bench_cuda_gemm.py

Vector Add
++++++++++
"""

import itertools
import numpy
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas import DataFrame
from teachcompute.ext_test_case import measure_time, unit_test_going
import torch

has_cuda = torch.cuda.is_available()

try:
    from teachcompute.validation.cuda.cuda_gemm import (
        matmul_v1_cuda,
        matmul_v2_cuda,
        matmul_v3_cuda,
    )
except ImportError:
    has_cuda = False


def torch_matmul(m1, m2, r, trans_a, trans_b):
    torch.cuda.nvtx.range_push(
        f"torch_matmul, tA={1 if trans_a else 0}, tB={1 if trans_b else 0}"
    )
    if trans_a:
        if trans_b:
            r += m1.T @ m2.T
        else:
            r += m1.T @ m2
    elif trans_b:
        r += m1 @ m2.T
    else:
        r += m1 @ m2
    torch.cuda.nvtx.range_pop()


def matmul_v1(t1, t2, r, trans_a, trans_b):
    torch.cuda.nvtx.range_push(
        f"matmul_v1, tA={1 if trans_a else 0}, tB={1 if trans_b else 0}"
    )
    matmul_v1_cuda(
        *t1.shape, t1.data_ptr(), *t2.shape, t2.data_ptr(), r.data_ptr(), True, True
    )
    torch.cuda.nvtx.range_pop()


def matmul_v2(t1, t2, r, trans_a, trans_b):
    torch.cuda.nvtx.range_push(
        f"matmul_v2, tA={1 if trans_a else 0}, tB={1 if trans_b else 0}"
    )
    matmul_v2_cuda(
        *t1.shape, t1.data_ptr(), *t2.shape, t2.data_ptr(), r.data_ptr(), True, True
    )
    torch.cuda.nvtx.range_pop()


def matmul_v3(t1, t2, r, trans_a, trans_b):
    torch.cuda.nvtx.range_push(
        f"matmul_v3, tA={1 if trans_a else 0}, tB={1 if trans_b else 0}"
    )
    matmul_v3_cuda(
        *t1.shape, t1.data_ptr(), *t2.shape, t2.data_ptr(), r.data_ptr(), True, True
    )
    torch.cuda.nvtx.range_pop()


fcts = [torch_matmul, matmul_v1, matmul_v2, matmul_v3]

obs = []
dims = [2**9, 2**10]  # , 2**11]
if unit_test_going():
    dims = [16, 32, 64]
for trans_a, trans_b, dim, fct in tqdm(
    list(itertools.product([False, True], [False, True], dims, fcts))
):
    repeat, number = (10, 10) if dim <= 2**10 else (5, 5)
    values = numpy.ones((dim, dim), dtype=numpy.float32) / (dim * repeat * number)
    t1 = torch.Tensor(values).to("cuda:0")
    t2 = torch.Tensor(values).to("cuda:0")
    r = torch.zeros(t1.shape).to("cuda:0")

    if has_cuda:

        # warmup
        for _ in range(3):
            fct(t1, t2, r, trans_a=trans_a, trans_b=trans_b)
        r = torch.zeros(t1.shape).to("cuda:0")
        res = measure_time(
            lambda fct=fct, t1=t1, t2=t2, r=r, trans_a=trans_a, trans_b=trans_b: fct(
                t1, t2, r, trans_a=trans_a, trans_b=trans_b
            ),
            repeat=repeat,
            number=number,
            div_by_number=True,
        )

        res.update(
            dict(
                dim=dim,
                shape=tuple(values.shape),
                fct=fct.__name__,
                tA=trans_a,
                tB=trans_b,
                tt=f"tt{1 if trans_a else 0}{1 if trans_b else 0}",
            )
        )
        obs.append(res)


if has_cuda:
    df = DataFrame(obs)
    df.to_csv("plot_bench_cuda_gemm.csv", index=False)
    df.to_excel("plot_bench_cuda_gemm.xlsx", index=False)
    print(df.head())


##############################################
# Plots
# +++++

if has_cuda:
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for tt in ["tt00", "tt01", "tt10", "tt11"]:
        piv_time = df[df.tt == tt].pivot(index="dim", columns="fct", values="average")
        a = ax[int(tt[2]), int(tt[3])]
        piv_time.plot(ax=a, logx=True, title=f"tA,tB={tt}")
        cb = piv_time["torch_matmul"].astype(float).copy()
        for c in piv_time.columns:
            piv_time[c] = cb / piv_time[c].astype(float)
        print(f"speed up for tt={tt}")
        print(piv_time)
        print()
    fig.suptitle("greater is better")
    fig.tight_layout()
    fig.savefig("plot_bench_cuda_gemm.png")

##############################################
#
