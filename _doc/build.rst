
=====
Build
=====

Spark
=====

Notebooks using :epkg:`pyspark` can be run locally.
They are disabled on CI but they can be run as well with the
rest of the other unit tests with::

    TEST_SPARSE=1 pytest _unittests

Build from source
=================

The packages relies on :epkg:`cmake` to build the C++ extensions.
whether it wrapped with :epkg:`pybind11` or :epkg:`cython`.
Both options are available and can be linked with :epkg:`openmp`,
:epkg:`eigen`, :epkg:`CUDA`.
*cmake* is called from `setup.py
<https://github.com/sdpython/teachcompute/blob/main/setup.py#L198>`_
with two instructions:

* ``python setup.py build_ext --inplace``, the legacy way
* ``pip install -e .``, the new way

By default, *cmake* builds with CUDA if it is available. It can be disabled:

* ``python setup.py build_ext -v --inplace --with-cuda=0``, the legacy way
* ``pip install -e . -v --config-settings="--with-cuda=0"``, the new way (not fully working yet)
* ``pip install -e . -v --global-option "--with-cuda=0"``, the deprecated way
* ``USE_CUDA=0 pip install -e . -v``, the run around way

In case there are multiple versions of CUDA installed, option `cuda-version`
can be specified:

::

    python setup.py build_ext --inplace --cuda-version=11.8

.. toctree::
    :maxdepth: 1    
    
    build_cython
    build_pybind11
    build_cuda
