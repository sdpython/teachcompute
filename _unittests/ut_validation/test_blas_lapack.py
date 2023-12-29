import unittest
import numpy as np
from teachcompute.ext_test_case import ExtTestCase
from teachcompute.validation.cython.blas_lapack import gemm_dot, pygemm


class TestBlasLapck(ExtTestCase):
    def test_pygemm_square(self):
        a = np.random.randint(10, size=(3, 3))
        b = np.random.randint(10, size=(3, 3))
        c = np.random.randint(10, size=(3, 3))
        for dtype in [np.float32, np.float64]:
            for ta in [True, False]:
                for tb in [True, False]:
                    a2 = a.astype(dtype).T.ravel()
                    b2 = b.astype(dtype).T.ravel()
                    c2 = c.T.astype(dtype).ravel()
                    pygemm(ta, tb, 3, 3, 3, 1.0, a2, 3, b2, 3, 1.0, c2, 3)
                    aa = a.T if ta else a
                    bb = b.T if tb else b
                    self.assertEqualArray(
                        (aa @ bb + c).astype(dtype), c2.reshape(c.shape).T
                    )

    def test_gemm_dot_square(self):
        a = np.random.randint(10, size=(3, 3))
        b = np.random.randint(10, size=(3, 3))
        for dtype in [np.float32, np.float64]:
            for ta in [True, False]:
                for tb in [True, False]:
                    a2 = a.astype(dtype)
                    b2 = b.astype(dtype)
                    got = gemm_dot(a2, b2, ta, tb)
                    aa = a.T if ta else a
                    bb = b.T if tb else b
                    self.assertEqualArray((aa @ bb).astype(dtype), got)

    def test_gemm_dot1(self):
        a = np.random.randint(10, size=(4, 3))
        b = np.random.randint(10, size=(3, 4))
        for dtype in [np.float32, np.float64]:
            for ta in [True, False]:
                for tb in [True, False]:
                    aa = a.T if ta else a
                    bb = b.T if tb else b
                    try:
                        expected = aa @ bb
                    except ValueError:
                        continue
                    a2 = a.astype(dtype)
                    b2 = b.astype(dtype)
                    got = gemm_dot(a2, b2, ta, tb)
                    self.assertEqualArray(expected.astype(dtype), got)

    def test_gemm_dot2(self):
        a = np.random.randint(10, size=(3, 4))
        b = np.random.randint(10, size=(3, 4))
        for dtype in [np.float32, np.float64]:
            for ta in [True, False]:
                for tb in [True, False]:
                    aa = a.T if ta else a
                    bb = b.T if tb else b
                    try:
                        expected = aa @ bb
                    except ValueError:
                        continue
                    a2 = a.astype(dtype)
                    b2 = b.astype(dtype)
                    got = gemm_dot(a2, b2, ta, tb)
                    self.assertEqualArray(expected.astype(dtype), got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
