import unittest
import numpy
from numba import njit, prange
from teachcompute.ext_test_case import ExtTestCase
from teachcompute.fctmr.fast_parallel_fctmr import fast_parallel_mapper


@njit(parallel=False, nopython=True, nogil=True)
def _mapcustom(input, output):
    for i in prange(0, len(input)):
        output[i] = input[i] + 1


def map_custom(input):
    output = numpy.empty(input.shape[0], dtype=input.dtype)
    _mapcustom(input, output)
    return output


class TestParallelFctMr(ExtTestCase):
    def test_parallel_mapper_sig(self):
        def func2(x):
            return x + 1

        li = numpy.array(list(range(0, 1000000)), dtype=numpy.float64)
        out = list(fast_parallel_mapper(func2, li, nogil=True, sigin="f8", sigout="f8"))
        self.assertEqual(out[:10], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        out = map_custom(li)
        self.assertEqualArray(
            out[:10], numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        )
        out = list(map(func2, li))
        self.assertEqual(out[:10], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

    def test_parallel_mapper_nosig(self):
        def func2(x):
            return x + 1

        li = numpy.array(list(range(0, 1000000)), dtype=numpy.float64)
        out = list(fast_parallel_mapper(func2, li, nogil=False, nopython=False))
        self.assertEqual(out[:10], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        out = map_custom(li)
        self.assertEqualArray(
            out[:10], numpy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        )
        out = list(map(func2, li))
        self.assertEqual(out[:10], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


if __name__ == "__main__":
    unittest.main()
