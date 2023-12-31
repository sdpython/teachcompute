Parallelization of a vector sum with C++
========================================

The sum of all elements in a vector is usully called a **reduced sum**
as it consists into the summation of all elements in a single float.
It may be simple as a computation examples but it is not simple to
parallelize.

.. toctree::
    :caption: Map Reduce
    :maxdepth: 1
    
    auto_examples/plot_bench_cpu_vector_sum
    auto_examples/plot_bench_cpu_vector_sum_parallel
    auto_examples/plot_bench_cpu_vector_sum_avx_parallel
