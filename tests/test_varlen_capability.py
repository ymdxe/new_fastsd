import types
import unittest

try:
    from src.kvcache_varlen import supports_varlen_model
    _DEPS = True
except Exception:  # pragma: no cover - dependency gate
    _DEPS = False


@unittest.skipUnless(_DEPS, "Torch/Transformers are required for varlen capability tests")
class VarlenCapabilityTests(unittest.TestCase):
    def test_non_qwen3_model_is_rejected(self):
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="llama"),
            model=types.SimpleNamespace(layers=[]),
        )
        self.assertFalse(supports_varlen_model(model))

    def test_qwen3_shape_is_accepted(self):
        attention = types.SimpleNamespace(q_norm=object(), k_norm=object())
        layer = types.SimpleNamespace(self_attn=attention)
        model = types.SimpleNamespace(
            config=types.SimpleNamespace(model_type="qwen3"),
            model=types.SimpleNamespace(layers=[layer]),
        )
        self.assertTrue(supports_varlen_model(model))


if __name__ == "__main__":
    unittest.main()
