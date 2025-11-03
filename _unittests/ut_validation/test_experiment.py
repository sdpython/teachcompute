import unittest
import numpy
from teachcompute.ext_test_case import ExtTestCase
from teachcompute import has_cuda

if has_cuda():
    from teachcompute.validation.cuda.cuda_example_py import measure_vector_add_half
else:
    measure_vector_add_half = None


class TestExperiment(ExtTestCase):
    @unittest.skipIf(measure_vector_add_half is None, reason="CUDA not available")
    def test_vector_add_cuda(self):
        v1 = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.uint16)
        v2 = numpy.array([[10, 1, 4, 5, 6, 7]], dtype=numpy.uint16)
        res = measure_vector_add_half(v1, v2, 0, repeat=10)
        self.assertIsInstance(res, dict)
        self.assertEqual(len(res), 3)

        if __name__ == "__main__":
            repeat = 100
            for _ in range(3):
                for size in [1024, 2048, 4096, 8192, 65536, 2**20, 2**25]:
                    v1 = numpy.random.randint(0, 1024, size=(size,)).astype(
                        numpy.uint16
                    )
                    v2 = numpy.random.randint(0, 1024, size=(size,)).astype(
                        numpy.uint16
                    )
                    self.assertEqual(v1.size, size)
                    res = measure_vector_add_half(v1, v2, 0, repeat=repeat)
                    ratio = res["half2"] / res["half"] - 1
                    print(
                        f"size={size} repeat={repeat} half={res['half'] / repeat:.4g} "
                        f"half2={res['half2'] / repeat:.4g} ratio={ratio:.3f}"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
