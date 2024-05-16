import unittest
import numpy as np
from teachcompute.ext_test_case import ExtTestCase
from teachcompute import has_cuda

if has_cuda():
    from teachcompute.validation.cuda.cuda_gemm import matmul_v1_cuda
else:
    matmul_v1_cuda = None


class TestGemmCuda(ExtTestCase):
    @unittest.skipIf(matmul_v1_cuda is None, reason="CUDA not available")
    def test_matmul_v1(self):
        import torch

        t1 = torch.rand(32, 64, dtype=torch.float32).to("cuda:0")
        t2 = torch.rand(64, 128, dtype=torch.float32).to("cuda:0")
        tt = t1 @ t2
        tt_np = tt.detach().cpu().numpy()
        self.assertEqual((32, 128), tuple(tt_np.shape))
        res = torch.zeros(*tt.shape, dtype=t1.dtype).to("cuda:0")
        res_np = res.detach().cpu().numpy()
        expected = np.zeros(res_np.shape).astype(np.float32)
        self.assertEqualArray(expected, res_np)
        matmul_v1_cuda(
            *t1.shape, t1.data_ptr(), *t2.shape, t2.data_ptr(), res.data_ptr()
        )
        res_np = res.detach().cpu().numpy()
        self.assertEqualArray(tt_np, res_np, atol=1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
