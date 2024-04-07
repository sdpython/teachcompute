import unittest
from teachcompute.ext_test_case import ExtTestCase, skipif_ci_apple


class TestPhi(ExtTestCase):

    @skipif_ci_apple("crash")
    def test_get_phi_model_mask_eager(self):
        from teachcompute.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(
            _attn_implementation="eager", with_mask=True
        )
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)

    @skipif_ci_apple("crash")
    def test_get_phi_model_nomask_eager(self):
        from teachcompute.torch_models.phi_helper import (
            get_phi_model,
        )

        model, model_inputs = get_phi_model(
            _attn_implementation="eager", with_mask=False
        )
        self.assertEqual(len(model_inputs[0]), 1)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
