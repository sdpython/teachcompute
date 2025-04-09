import unittest
import numpy
from teachcompute.ext_test_case import ExtTestCase, skipif_ci_windows, skipif_ci_apple


class TestExampleThreads(ExtTestCase):
    def test_test_sum_no_mutex(self):
        from teachcompute.validation.cpu._validation import test_sum_no_mutex

        self.assertEqualLess(test_sum_no_mutex(10), 10)

if __name__ == "__main__":
    unittest.main(verbosity=2)
