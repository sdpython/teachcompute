import os
import unittest
from teachcompute.ext_test_case import ExtTestCase
from teachcompute.datasets import mortality_table


class TestEuroStat(ExtTestCase):
    def test_mortalite_euro_stat(self):
        outfile = mortality_table(to=os.path.dirname(__file__), stop_at=100)
        self.assertExists(outfile)


if __name__ == "__main__":
    unittest.main(verbosity=2)
