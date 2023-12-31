
.. |gitlogo| image:: _static/git_logo.png
             :height: 20

===============
Calcul intensif
===============

Un seul ordinateur ne suffit plus aujourd'hui pour satisfaire tous les besoins
des datascientists. Ils ont besoin de nombreuses machines pour traiter
des volumns gigantesques de données. Ils ont besoin d'écrire du code
efficace pour apprendre des modèles de machines learning de plus en plus
gros. Calculs des statistiques simples sur des données de plusieurs centaines
de gigaoctets se fait le plus souvent avec une technologique Map/Reduce
sur des clusters de machine. Apprendre des réseaux de neurones profonds
se fait le plus souvent avec des processeurs :epkg:`GPU` et de façon
parallélisée. Ce site `GitHub/teachcompute <https://github.com/sdpython/teachcompute>`_ |gitlogo|
introduit ces deux voies.

.. toctree::
    :maxdepth: 1
    :caption: Lectures

    introduction
    build
    articles/index

.. toctree::
    :maxdepth: 1
    :caption: Exercices

    practice/index_expose
    practice/index_mapreduce
    practice/index_spark

.. toctree::
    :maxdepth: 1
    :caption: Compléments

    i_index
    license
    CHANGELOGS

L'intelligence artificielle est entrée dans le quotidien.
Machine learning, deep learning, la porte d'entrée se fait
par la programmation et principalement avec le langgage python.
Le site `Xavier Dupré <http://www.xavierdupre.fr/>`_
contient beaucoup d'exemples sur beaucoup de sujets,
souvent reliés au machine learning.

.. image:: https://dev.azure.com/xavierdupre3/teachcompute/_apis/build/status%2Fsdpython.teachcompute?branchName=main
    :target: https://dev.azure.com/xavierdupre3/teachcompute/_build/latest?definitionId=29&branchName=main

.. image:: https://ci.appveyor.com/api/projects/status/fl1sge2kumhg8v51?svg=true
    :target: https://ci.appveyor.com/project/sdpython/teachcompute
    :alt: Build Status Windows

.. image:: https://badge.fury.io/py/teachcompute.svg
    :target: https://pypi.org/project/teachcompute/

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
    :alt: MIT License
    :target: https://opensource.org/licenses/MIT

.. image:: https://codecov.io/github/sdpython/teachcompute/branch/main/graph/badge.svg?token=zmROB7lJAt 
    :target: https://codecov.io/github/sdpython/teachcompute

.. image:: http://img.shields.io/github/issues/sdpython/teachcompute.svg
    :alt: GitHub Issues
    :target: https://github.com/sdpython/teachcompute/issues

.. image:: https://img.shields.io/github/repo-size/sdpython/teachcompute
    :target: https://github.com/sdpython/teachcompute/
    :alt: size

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

Older versions
++++++++++++++

* `0.1.0 <../v0.1.0/index.html>`_
