import unittest
from python1.operations import running_mean


class TestRunningMean(unittest.TestCase):
    def test_single_element(self):
        self.assertEqual(running_mean([5]), [5.0])

    def test_integers(self):
        result = running_mean([1, 2, 3, 4])
        self.assertEqual(result, [1.0, 1.5, 2.0, 2.5])

    def test_floats(self):
        result = running_mean([0.0, 1.0, 2.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0)

    def test_negative_values(self):
        result = running_mean([-1, -2, -3])
        self.assertAlmostEqual(result[0], -1.0)
        self.assertAlmostEqual(result[1], -1.5)
        self.assertAlmostEqual(result[2], -2.0)

    def test_length_preserved(self):
        data = list(range(10))
        self.assertEqual(len(running_mean(data)), len(data))

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            running_mean([])

    def test_last_value_is_mean(self):
        data = [1, 3, 5, 7, 9]
        result = running_mean(data)
        expected_last = sum(data) / len(data)
        self.assertAlmostEqual(result[-1], expected_last)


if __name__ == "__main__":
    unittest.main(verbosity=2)
