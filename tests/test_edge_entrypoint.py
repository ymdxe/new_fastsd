import importlib.util
import py_compile
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
EDGE_PATH = REPO_ROOT / "edge" / "edge.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for attribute_name, value in attributes.items():
        setattr(module, attribute_name, value)
    return module


def _load_edge_with_auto_gptq_blocked():
    """Import edge.py with lightweight dependency stubs and no auto_gptq."""

    class FakeAutoModelForCausalLM:
        calls = []

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            cls.calls.append((args, kwargs))

            class LoadedModel:
                def eval(self):
                    return self

            return LoadedModel()

    class FakeDecoding:
        pass

    fake_torch = _module("torch", Tensor=object, long=object)
    fake_torch.no_grad = lambda function=None: function or (lambda wrapped: wrapped)
    fake_src = _module("src")
    fake_src.__path__ = []
    stubs = {
        "torch": fake_torch,
        "requests": _module("requests", Session=object),
        "transformers": _module(
            "transformers", AutoModelForCausalLM=FakeAutoModelForCausalLM
        ),
        "src": fake_src,
        "src.arrival": _module(
            "src.arrival",
            poisson_arrival_offsets=lambda *args, **kwargs: [],
            shard_samples=lambda samples, *args, **kwargs: samples,
        ),
        "src.engine": _module("src.engine", Decoding=FakeDecoding),
        "src.kvcache": _module("src.kvcache", KVCacheModel=object),
        "src.metrics": _module(
            "src.metrics", elapsed_ms=lambda *args, **kwargs: 0.0, tpot_ms=lambda *args, **kwargs: 0.0
        ),
        "src.runtime": _module(
            "src.runtime",
            configure_torch_threads=lambda *args, **kwargs: None,
            resolve_dtype=lambda *args, **kwargs: "float32",
        ),
        "src.util": _module(
            "src.util",
            parse_arguments=lambda *args, **kwargs: None,
            seed_everything=lambda *args, **kwargs: None,
        ),
        # A None entry makes any attempted top-level import fail immediately.
        "auto_gptq": None,
    }
    module_name = "edge_optional_dependency_test"
    spec = importlib.util.spec_from_file_location(module_name, EDGE_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)
    return module, FakeAutoModelForCausalLM


class EdgeEntrypointTests(unittest.TestCase):
    def test_edge_module_compiles(self):
        py_compile.compile(str(EDGE_PATH), doraise=True)

    def test_mt_bench_uses_chat_template(self):
        source = EDGE_PATH.read_text(encoding="utf-8")
        self.assertIn('self.args.dataset != "mt_bench"', source)
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn("configure_torch_threads", source)
        self.assertIn("resolve_dtype", source)
        self.assertIn("if torch.cuda.is_available()", source)

    def test_cpu_non_gptq_import_and_load_do_not_require_auto_gptq(self):
        edge_module, fake_auto_model = _load_edge_with_auto_gptq_blocked()
        runner = object.__new__(edge_module.EdgeRunner)
        runner.args = types.SimpleNamespace(draft_dtype="float32")

        model_dir = str(REPO_ROOT / "non-gptq-test-model")
        loaded_model = runner._load_draft_model(model_dir, "cpu")

        self.assertIsNotNone(loaded_model)
        self.assertEqual(len(fake_auto_model.calls), 1)
        self.assertEqual(fake_auto_model.calls[0][1]["device_map"], {"": "cpu"})

    def test_gptq_request_reports_missing_auto_gptq_clearly(self):
        edge_module, _ = _load_edge_with_auto_gptq_blocked()
        runner = object.__new__(edge_module.EdgeRunner)
        runner.args = types.SimpleNamespace(draft_dtype="float32")

        model_dir = str(REPO_ROOT / "gptq-test-model")
        with patch.dict(sys.modules, {"auto_gptq": None}):
            with patch.object(edge_module.os.path, "exists", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "GPTQ draft requested.*auto_gptq"):
                    runner._load_draft_model(model_dir, "cpu")


if __name__ == "__main__":
    unittest.main()
