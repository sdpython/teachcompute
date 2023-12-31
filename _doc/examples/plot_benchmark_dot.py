"""

.. _l-example-dot:

Compares dot implementations (numpy, python, blas)
==================================================

:epkg:`numpy` has a very fast implementation of
the dot product. It is difficult to be better and very easy
to be slower. This example looks into a couple of slower
implementations.

Compared implementations:

* :func:`pydot <teachcompute.validation.cython.dotpy.pydot>`
* `cblas_ddot <https://github.com/sdpython/teachcompute/
  blob/master/teachcompute/validation/cython/dot_blas_lapack.pyx#L39>`_

"""

import pprint
import numpy
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from teachcompute.validation.cython.dotpy import pydot
from teachcompute.validation.cython.dot_blas_lapack import cblas_ddot
from teachcompute.ext_test_case import measure_time_dim

##############################
# python dot: pydot
# +++++++++++++++++
#
# The first function :func:`pydot
# <teachcompute.validation.cython.pydot>` uses
# python to implement the dot product.

ctxs = [
    dict(
        va=numpy.random.randn(n).astype(numpy.float64),
        vb=numpy.random.randn(n).astype(numpy.float64),
        pydot=pydot,
        x_name=n,
    )
    for n in range(10, 1000, 100)
]

res_pydot = list(measure_time_dim("pydot(va, vb)", ctxs, verbose=1))

pprint.pprint(res_pydot[:2])

##############################
# numpy dot
# +++++++++
#

ctxs = [
    dict(
        va=numpy.random.randn(n).astype(numpy.float64),
        vb=numpy.random.randn(n).astype(numpy.float64),
        dot=numpy.dot,
        x_name=n,
    )
    for n in range(10, 50000, 100)
]

res_dot = list(measure_time_dim("dot(va, vb)", ctxs, verbose=1))

pprint.pprint(res_dot[:2])

##############################
# blas dot
# ++++++++
#
# :epkg:`numpy` implementation uses :epkg:`BLAS`.
# Let's make a direct call to it.

for ctx in ctxs:
    ctx["ddot"] = cblas_ddot

res_ddot = list(measure_time_dim("ddot(va, vb)", ctxs, verbose=1))

pprint.pprint(res_ddot[:2])

#############################
# Let's display the results
# +++++++++++++++++++++++++

df1 = DataFrame(res_pydot)
df1["fct"] = "pydot"
df2 = DataFrame(res_dot)
df2["fct"] = "numpy.dot"
df3 = DataFrame(res_ddot)
df3["fct"] = "ddot"

cc = concat([df1, df2, df3])
cc["N"] = cc["x_name"]

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cc[cc.N <= 1100].pivot(index="N", columns="fct", values="average").plot(
    logy=True, logx=True, ax=ax[0]
)
cc[cc.fct != "pydot"].pivot(index="N", columns="fct", values="average").plot(
    logy=True, logx=True, ax=ax[1]
)
ax[0].set_title("Comparison of dot implementations")
ax[1].set_title("Comparison of dot implementations\nwithout python")

#################################
# The results depends on the machine, its
# number of cores, the compilation settings
# of :epkg:`numpy` or this module.
