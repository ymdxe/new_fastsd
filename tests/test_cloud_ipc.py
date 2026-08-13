import py_compile
from pathlib import Path
import unittest


class CloudIPCPayloadTests(unittest.TestCase):
    def test_http_queue_payloads_do_not_send_torch_storages(self):
        repo = Path(__file__).resolve().parents[1]
        cloud_path = repo / "cloud" / "cloud_service.py"
        engine_path = repo / "src" / "engine.py"
        cloud_text = cloud_path.read_text(encoding="utf-8")
        engine_text = engine_path.read_text(encoding="utf-8")

        self.assertNotIn("torch.tensor(req.draft_output", cloud_text)
        self.assertGreaterEqual(
            cloud_text.count('"draft_output": [int(token) for token in req.draft_output]'),
            2,
        )
        self.assertNotIn('"final_token": new_token', engine_text)
        self.assertIn("if not torch.is_tensor(draft_output):", engine_text)

        py_compile.compile(str(cloud_path), doraise=True)
        py_compile.compile(str(engine_path), doraise=True)


if __name__ == "__main__":
    unittest.main()
