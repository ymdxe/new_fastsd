import py_compile
from pathlib import Path
import unittest


class EdgeEntrypointTests(unittest.TestCase):
    def test_edge_module_compiles(self):
        repo = Path(__file__).resolve().parents[1]
        py_compile.compile(str(repo / "edge" / "edge.py"), doraise=True)

    def test_mt_bench_uses_chat_template(self):
        repo = Path(__file__).resolve().parents[1]
        source = (repo / "edge" / "edge.py").read_text(encoding="utf-8")
        self.assertIn('self.args.dataset != "mt_bench"', source)
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn("configure_torch_threads", source)
        self.assertIn("resolve_dtype", source)
        self.assertIn("if torch.cuda.is_available()", source)


if __name__ == "__main__":
    unittest.main()
