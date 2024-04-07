import unittest
from teachcompute.ext_test_case import ExtTestCase


class TestLlama(ExtTestCase):

    def test_get_llama_model_mask_sdpa(self):
        from teachcompute.torch_models.llama_helper import (
            get_llama_model,
        )

        model, model_inputs = get_llama_model(
            _attn_implementation="sdpa", with_mask=True
        )
        self.assertEqual(len(model_inputs[0]), 2)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)

    def test_get_llama_model_nomask_sdpa(self):
        from teachcompute.torch_models.llama_helper import (
            get_llama_model,
        )

        model, model_inputs = get_llama_model(
            _attn_implementation="sdpa", with_mask=False
        )
        self.assertEqual(len(model_inputs[0]), 1)
        expected = model(*model_inputs[0])
        self.assertNotEmpty(expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
