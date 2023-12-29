import unittest
from teachcompute.ext_test_case import ExtTestCase, ignore_warnings
from teachcompute import log


class TestLog(ExtTestCase):
    @ignore_warnings(UserWarning)
    def test_log(self):
        _, out, _ = self.capture(lambda: log(1, lambda: "GG"))
        self.assertEqual(out, "GG\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
