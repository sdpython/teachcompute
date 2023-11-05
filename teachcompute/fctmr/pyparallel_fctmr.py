"""
:epkg:`joblib` uses a module not documented
in the official :epkg:`Python` documentation:
`Python's undocumented ThreadPool
<https://lucasb.eyer.be/snips/python-thread-pool.html>`_.
"""
from typing import Callable, Iterable, Optional
from multiprocessing.pool import ThreadPool


def pyparallel_mapper(
    fct: Callable, gen: Iterable, threads: Optional[int] = None
) -> Iterable:
    """
    Applies function *fct* to a generator.
    Relies on *ThreadPool*.

    :param fct: function
    :param gen: generator
    :param threads: number of threads
    :return: generator

    If the number of threads is None,
    it is replaced by ``os.cpu_count() or 1``
    (see *multiprocessing.pool*).

    .. exref::
        :title: mapper
        :tag: progfonc

        .. runpython::
            :showcode:

            from teachcompute.fctmr.pyparallel_fctmr import pyparallel_mapper

            res = pyparallel_mapper(lambda x: x + 1, [4, 5])
            print(list(res))

    Unfortunately, the parallelization is not following
    the map/reduce concept in a sense that the function
    generates an intermediate list and creates an iterator
    on it.
    """
    pool = ThreadPool(processes=threads)
    return iter(pool.map(fct, gen))
