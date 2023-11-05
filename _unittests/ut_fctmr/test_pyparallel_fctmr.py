import unittest
from teachcompute.ext_test_case import ExtTestCase
from teachcompute.fctmr.pyparallel_fctmr import pyparallel_mapper


class TestPyParallelFctMr(ExtTestCase):
    def test_pyparallel_mapper(self):
        res = pyparallel_mapper(lambda x: x + 1, [4, 5])
        self.assertNotIsInstance(res, list)
        self.assertEqual(list(res), [5, 6])


if __name__ == "__main__":
    unittest.main()
