"""
Fonctions de calcul numérique en Python pur.
"""

from typing import List, Union

Number = Union[int, float]


def running_mean(data: List[Number]) -> List[float]:
    """
    Calcule la moyenne cumulée d'une séquence de nombres.

    Pour chaque position ``i``, la valeur retournée est la moyenne
    des éléments ``data[0], data[1], ..., data[i]``.

    :param data: séquence de nombres (liste d'entiers ou de flottants).
    :return: liste de flottants de même longueur que *data* contenant
        les moyennes cumulées.
    :raises ValueError: si *data* est vide.

    Exemple::

        >>> from python1.operations import running_mean
        >>> running_mean([1, 2, 3, 4])
        [1.0, 1.5, 2.0, 2.5]
    """
    if len(data) == 0:
        raise ValueError("data ne doit pas être vide.")
    result: List[float] = []
    total = 0.0
    for i, value in enumerate(data):
        total += value
        result.append(total / (i + 1))
    return result
