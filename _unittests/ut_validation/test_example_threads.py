import unittest
from teachcompute.ext_test_case import ExtTestCase


class TestExampleThreads(ExtTestCase):
    def test_test_sum_no_mutex(self):
        from teachcompute.validation.cpu._validation import test_sum_no_mutex

        self.assertLess(test_sum_no_mutex(10), 11)
        # self.assertLess(test_sum_no_mutex(1000000), 1000000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
