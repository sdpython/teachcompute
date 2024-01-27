"""
Simple *mapper* and *reducer* implemented in :epkg:`Python`
"""

from typing import Callable, Iterable
from itertools import groupby


def mapper(fct: Callable, gen: Iterable) -> Iterable:
    """
    Applies function *fct* to a generator.

    :param fct: function
    :param gen: generator
    :return: generator

    .. exref::
        :title: mapper
        :tag: progfonc

        .. runpython::
            :showcode:

            from teachcompute.fctmr import mapper

            res = mapper(lambda x: x + 1, [4, 5])
            print(list(res))

    .. faqref::
        :title: Différence entre un itérateur et un générateur ?
        :tag: faqprogfonc

        Un :epkg:`itérateur` et un :epkg:`générateur` produisent
        tous deux des éléments issus d'un ensemble. La différence
        vient du fait que qu'un :epkg:`itérateur` parcourt les
        éléments d'un ensemble qui existe en mémoire. Un :epkg:`générateur`
        produit ou calcule des éléments d'un ensemble qui n'existe
        pas en mémoire. Par conséquent, parcourir deux fois un ensemble
        avec un itérateur a un coût en :math:`O(n)` alors que pour
        un générateur, il faut ajouter le calcul de l'élément une
        seconde fois. Le coût est imprévisible et parfois il est
        préférable de :epkg:`cacher` les éléments pour le parcourir
        plusieurs fois : cela revient à transformer un :epkg:`générateur`
        en :epkg:`itérateur`. Un générateur est souvent défini comme suit
        en :epkg:`Python` :

        .. runpython::
            :showcode:

            def generate(some_iterator):
                for el in some_iterator:
                    yield el

            g = generate([4, 5])
            print(list(g))
            print(g.__class__.__name__)
    """
    return map(fct, gen)


def take(gen: Iterable, count: int = 5, skip: int = 0) -> Iterable:
    """
    Skips and takes elements from a generator.

    :param gen: generator
    :param count: number of elements to consider
    :param skip: skip the first elements
    :return: generator

    .. exref::
        :title: take
        :tag: progfonc

        .. runpython::
            :showcode:

            from teachcompute.fctmr import take
            res = take([4, 5, 6, 7, 8, 9], 2, 2)
            print(list(res))
    """
    took = 0
    for i, el in enumerate(gen):
        if i < skip:
            continue
        if took >= count:
            break
        yield el
        took += 1


def ffilter(fct: Callable, gen: Iterable) -> Iterable:
    """
    Filters out elements from a generator.

    :param fct: function
    :param gen: generator
    :return: generator

    .. exref::
        :title: filter
        :tag: progfonc

        .. runpython::
            :showcode:

            from teachcompute.fctmr import ffilter

            res = ffilter(lambda x: x % 2 == 0, [4, 5])
            print(list(res))
    """
    return filter(fct, gen)


def reducer(
    fctkey: Callable, gen: Iterable, asiter: bool = True, sort: bool = True
) -> Iterable:
    """
    Implements a reducer.

    :param fctkey: function which returns the key
    :param gen: generator
    :param asiter: returns an iterator on each element of the group
        of the group itself
    :param sort: sort elements by key before grouping
    :return: generator

    .. exref::
        :title: reducer
        :tag: progfonc

        .. runpython::
            :showcode:

            from teachcompute.fctmr import reducer
            res = reducer(lambda x: x[0], [
                          ('a', 1), ('b', 2), ('a', 3)], asiter=False)
            print(list(res))
    """
    if sort:
        new_gen = map(lambda x: x[1], sorted(map(lambda el: (fctkey(el), el), gen)))
        gr = groupby(new_gen, fctkey)
    else:
        gr = groupby(gen, fctkey)
    if asiter:
        # Cannot return gr. Python is confused when yield and return
        # are used in the same function.
        for _ in gr:
            yield _
    else:
        for key, it in gr:
            yield key, list(it)


def combiner(
    fctkey1: Callable,
    gen1: Iterable,
    fctkey2: Callable,
    gen2: Iterable,
    how: str = "inner",
) -> Iterable:
    """
    Joins (or combines) two generators.
    The function is written based on two reducers.
    The function is more efficient if the groups
    of the second ensemble *gen2* are shorter
    as each of them will be held in memory.

    :param fctkey1: function which returns the key for gen1
    :param gen1: generator for the first element
    :param fctkey2: function which returns the key for gen2
    :param gen2: generator for the second element
    :param how: *inner*, *outer*, *left*, right*
    :return: generator

    .. exref::
        :title: combiner or join
        :tag: progfonc

        .. runpython::
            :showcode:

            from teachcompute.fctmr import combiner

            def c0(el):
                return el[0]

            ens1 = [('a', 1), ('b', 2), ('a', 3)]
            ens2 = [('a', 10), ('b', 20), ('a', 30)]
            res = combiner(c0, ens1, c0, ens2)
            print(list(res))
    """
    gr1 = reducer(fctkey1, gen1, asiter=True, sort=True)
    gr2 = reducer(fctkey2, gen2, asiter=False, sort=True)

    def fetch_next(it):
        "local function"
        try:
            return next(it)
        except StopIteration:
            return None, None

    k1, g1 = fetch_next(gr1)
    k2, g2 = fetch_next(gr2)
    while k1 is not None or k2 is not None:
        if k1 is None or (k2 is not None and k2 < k1):
            if how in ("outer", "right"):
                for el in g2:
                    yield None, el
                k2, g2 = fetch_next(gr2)
            else:
                break
        elif k2 is None or k1 < k2:
            if how in ("outer", "left"):
                for el in g1:
                    yield el, None
                k1, g1 = fetch_next(gr1)
            else:
                break
        elif k1 == k2:
            for el1 in g1:
                for el2 in g2:
                    yield el1, el2
            k1, g1 = fetch_next(gr1)
            k2, g2 = fetch_next(gr2)
