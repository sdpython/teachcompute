import unittest
import numpy as np
from teachcompute.ext_test_case import ExtTestCase
from teachcompute import has_cuda

if has_cuda():
    from teachcompute.validation.cuda.cuda_gemm import (
        matmul_v1_cuda,
        matmul_v2_cuda,
        matmul_v3_cuda,
    )
else:
    matmul_v1_cuda = None


class TestGemmCuda(ExtTestCase):

    @classmethod
    def _get_cuda_tensor(cls, *shape, dtype=None, rev=False):
        import torch

        n = np.prod(shape)
        val = np.arange(n) / n
        if rev:
            val[::2] -= 1
            val = np.abs(val)
        return torch.Tensor(val.reshape(shape)).to(dtype).to("cuda:0")

    @unittest.skipIf(matmul_v1_cuda is None, reason="CUDA not available")
    def test_matmul_v1_false_false(self):
        import torch

        t1 = self._get_cuda_tensor(96, 64, dtype=torch.float32)
        t2 = self._get_cuda_tensor(64, 128, dtype=torch.float32, rev=True)
        tt = t1 @ t2
        tt_np = tt.detach().cpu().numpy()
        self.assertEqual((96, 128), tuple(tt_np.shape))
        res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
        res_np = res.detach().cpu().numpy()
        expected = np.zeros(res_np.shape).astype(np.float32)
        self.assertEqualArray(expected, res_np)
        matmul_v1_cuda(
            *t1.shape, t1.data_ptr(), *t2.shape, t2.data_ptr(), res.data_ptr()
        )
        res_np = res.detach().cpu().numpy()
        self.assertEqualArray(tt_np, res_np, atol=1e-4)

    def test_matmul_v1_true_false(self):
        import torch

        t1 = self._get_cuda_tensor(64, 96, dtype=torch.float32)
        t2 = self._get_cuda_tensor(64, 128, dtype=torch.float32, rev=True)
        tt = (t1.T) @ t2
        tt_np = tt.detach().cpu().numpy()
        res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
        matmul_v1_cuda(
            *t1.shape,
            t1.data_ptr(),
            *t2.shape,
            t2.data_ptr(),
            res.data_ptr(),
            True,
            False,
        )
        res_np = res.detach().cpu().numpy()
        self.assertEqualArray(tt_np, res_np, atol=1e-5)

    def test_matmul_v1_false_true(self):
        import torch

        t1 = self._get_cuda_tensor(96, 64, dtype=torch.float32)
        t2 = self._get_cuda_tensor(128, 64, dtype=torch.float32, rev=True)
        tt = t1 @ t2.T
        tt_np = tt.detach().cpu().numpy()
        res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
        matmul_v1_cuda(
            *t1.shape,
            t1.data_ptr(),
            *t2.shape,
            t2.data_ptr(),
            res.data_ptr(),
            False,
            True,
        )
        res_np = res.detach().cpu().numpy()
        self.assertEqualArray(tt_np, res_np, atol=3e-5)

    def test_matmul_v1_true_true(self):
        import torch

        t1 = self._get_cuda_tensor(64, 96, dtype=torch.float32)
        t2 = self._get_cuda_tensor(128, 64, dtype=torch.float32, rev=True)
        tt = t1.T @ t2.T
        tt_np = tt.detach().cpu().numpy()
        res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
        matmul_v1_cuda(
            *t1.shape,
            t1.data_ptr(),
            *t2.shape,
            t2.data_ptr(),
            res.data_ptr(),
            True,
            True,
        )
        res_np = res.detach().cpu().numpy()
        self.assertEqualArray(tt_np, res_np, atol=3e-5)

    def test_matmul_v2(self):
        import torch

        for ta, tb in [(False, False), (False, True), (True, False), (True, True)]:
            sh1 = (64, 96) if ta else (96, 64)
            sh2 = (64, 128) if not tb else (128, 64)
            t1 = self._get_cuda_tensor(*sh1, dtype=torch.float32)
            t2 = self._get_cuda_tensor(*sh2, dtype=torch.float32, rev=True)
            t1_ = t1.T if ta else t1
            t2_ = t2.T if tb else t2
            tt = t1_ @ t2_
            tt_np = tt.detach().cpu().numpy()
            res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
            with self.subTest(ta=ta, tb=tb, sh1=sh1, sh2=sh2):
                matmul_v2_cuda(
                    *t1.shape,
                    t1.data_ptr(),
                    *t2.shape,
                    t2.data_ptr(),
                    res.data_ptr(),
                    ta,
                    tb,
                )
                res_np = res.detach().cpu().numpy()
                self.assertEqualArray(tt_np, res_np, atol=3e-5)

    def test_matmul_v3(self):
        import torch

        for ta, tb in [(False, False), (False, True), (True, False), (True, True)]:
            sh1 = (256, 384) if ta else (384, 256)
            sh2 = (256, 128) if not tb else (128, 256)
            t1 = self._get_cuda_tensor(*sh1, dtype=torch.float32)
            t2 = self._get_cuda_tensor(*sh2, dtype=torch.float32, rev=True)
            t1_ = t1.T if ta else t1
            t2_ = t2.T if tb else t2
            tt = t1_ @ t2_
            tt_np = tt.detach().cpu().numpy()
            res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
            with self.subTest(ta=ta, tb=tb, sh1=sh1, sh2=sh2):
                matmul_v3_cuda(
                    *t1.shape,
                    t1.data_ptr(),
                    *t2.shape,
                    t2.data_ptr(),
                    res.data_ptr(),
                    ta,
                    tb,
                )
                res_np = res.detach().cpu().numpy()
                self.assertEqualArray(tt_np, res_np, rtol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
