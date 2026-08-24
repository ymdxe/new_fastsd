"""Focused portability checks for scripts/diagnose_environment.sh."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_environment.sh"


class DiagnoseEnvironmentScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")

    def test_uses_git_1_8_compatible_branch_detection(self):
        self.assertNotIn("git branch --show-current", self.source)
        self.assertIn('git symbolic-ref HEAD', self.source)
        self.assertIn('git rev-parse --short HEAD', self.source)
        self.assertIn("detached (commit", self.source)

    def test_searches_both_server_model_roots_and_allows_override(self):
        self.assertIn('FASTSD_MODEL_ROOTS', self.source)
        self.assertIn('FASTSD_MODEL_ROOT', self.source)
        self.assertIn('add_model_root "/home/zhangh/models"', self.source)
        self.assertIn('add_model_root "/home/hdd/zhangh/models"', self.source)
        for model_name in ("Qwen3-1.7B", "Qwen3-8B", "Qwen3-0.6B"):
            self.assertIn(model_name, self.source)

    def test_reports_total_physical_cores_from_socket_product(self):
        self.assertIn('cores_per_socket="$(lscpu_value', self.source)
        self.assertIn('sockets="$(lscpu_value', self.source)
        self.assertIn("physical_cores=$((cores_per_socket * sockets))", self.source)
        self.assertIn("物理核心: $physical_cores", self.source)

    def test_shell_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable on this host")
        if os.name == "nt":
            self.skipTest("POSIX shell syntax check is run on Linux; Windows bash path mapping is not reliable")
        subprocess.run([bash, "-n", str(SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
