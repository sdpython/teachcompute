
=====
Build
=====

Spark
=====

Notebooks using :epkg:`pyspark` can be run locally.
They are disabled on CI but they can be run as well with the
rest of the other unit tests with::

    TEST_SPARSE=1 pytest _unittests
