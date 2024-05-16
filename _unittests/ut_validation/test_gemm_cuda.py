import unittest
from teachcompute.ext_test_case import ExtTestCase
from teachcompute import has_cuda

if has_cuda():
    from teachcompute.validation.cuda.cuda_gemm import matmul_v1
else:
    matmul_v1 = None


class TestGemmCuda(ExtTestCase):
    @unittest.skipIf(matmul_v1 is None, reason="CUDA not available")
    def test_matmul_v1(self):
        import torch

        t1 = torch.rand(32, 64, dtype=torch.float32).to("cuda:0")
        t2 = torch.rand(64, 128, dtype=torch.float32).to("cuda:0")
        tt = t1 @ t2
        res = torch.empty(*tt.shape, dtype=t1.dtype).to("cuda:0")
        matmul_v1(*t1.size, t1.data_ptr(), *t2.size, t2.data_ptr(), res.data_ptr())
        self.assertEqualArray(tt.detach().cup().numpy(), res.detach().cup().numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
