python1 — package Python pur
==============================

Ce tutoriel montre comment structurer un **package Python pur** avec :

* une fonction documentée,
* des tests unitaires.

Structure
---------

.. code-block:: text

    python1/
    ├── python1/          # le package
    │   ├── __init__.py
    │   └── operations.py
    ├── test_python1.py   # tests unitaires
    └── setup.py

Installation
------------

Aucune compilation nécessaire (package Python pur) :

.. code-block:: bash

    pip install -e .

Utilisation
-----------

.. code-block:: python

    from python1 import running_mean

    data = [1, 2, 3, 4, 5]
    print(running_mean(data))
    # [1.0, 1.5, 2.0, 2.5, 3.0]

Lancer les tests
----------------

.. code-block:: bash

    python -m unittest test_python1 -v

Fonction disponible
-------------------

``running_mean(data)``
    Calcule la moyenne cumulée d'une séquence de nombres.
    Retourne une liste de flottants de même longueur que *data*.

    .. code-block:: python

        >>> from python1.operations import running_mean
        >>> running_mean([1, 2, 3, 4])
        [1.0, 1.5, 2.0, 2.5]
