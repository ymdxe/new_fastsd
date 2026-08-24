"""Focused portability checks for scripts/diagnose_environment.sh."""

import os
import shutil
import stat
import subprocess
import tempfile
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
        self.assertIn("if (key == wanted)", self.source)
        self.assertIn("lscpu_value 'Core(s) per socket'", self.source)
        self.assertIn("lscpu_value 'Socket(s)'", self.source)
        self.assertIn("lscpu_value 'NUMA node(s)'", self.source)
        self.assertNotIn(r"Core\(s\)", self.source)
        self.assertNotIn(r"Socket\(s\)", self.source)
        self.assertNotIn(r"NUMA node\(s\)", self.source)

    def test_mocked_lscpu_reports_total_physical_cores_without_awk_warnings(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable on this host")
        if os.name == "nt":
            self.skipTest("POSIX executable regression test is run on Linux")

        fixtures = (
            (20, 2, 40),
            (24, 4, 96),
        )
        for cores_per_socket, sockets, expected_physical_cores in fixtures:
            with self.subTest(cores_per_socket=cores_per_socket, sockets=sockets):
                with tempfile.TemporaryDirectory() as temp_dir:
                    mock_bin = Path(temp_dir) / "bin"
                    mock_bin.mkdir()
                    mock_lscpu = mock_bin / "lscpu"
                    fixture = "\n".join(
                        (
                            "Architecture:          x86_64",
                            "CPU(s):                "
                            + str(cores_per_socket * sockets * 2),
                            "Core(s) per socket:    " + str(cores_per_socket),
                            "Socket(s):             " + str(sockets),
                            "NUMA node(s):          " + str(sockets),
                        )
                    )
                    mock_lscpu.write_text(
                        "#!/bin/sh\n"
                        "cat <<'EOF'\n"
                        + fixture
                        + "\nEOF\n",
                        encoding="utf-8",
                    )
                    mock_lscpu.chmod(
                        mock_lscpu.stat().st_mode
                        | stat.S_IXUSR
                        | stat.S_IXGRP
                        | stat.S_IXOTH
                    )

                    environment = os.environ.copy()
                    environment["PATH"] = os.pathsep.join(
                        (str(mock_bin), environment.get("PATH", ""))
                    )
                    result = subprocess.run(
                        [bash, str(SCRIPT)],
                        cwd=SCRIPT.parents[1],
                        env=environment,
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        check=False,
                    )

                    self.assertEqual(
                        result.returncode,
                        0,
                        msg=result.stdout + "\n" + result.stderr,
                    )
                    self.assertIn(
                        f"物理核心: {expected_physical_cores} ",
                        result.stdout,
                    )
                    self.assertIn(f"Socket数: {sockets}", result.stdout)
                    self.assertIn(f"NUMA节点: {sockets}", result.stdout)
                    self.assertNotIn("物理核心: unknown", result.stdout)
                    self.assertNotIn("escape sequence", result.stderr)

    def test_shell_syntax(self):
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is unavailable on this host")
        if os.name == "nt":
            self.skipTest("POSIX shell syntax check is run on Linux; Windows bash path mapping is not reliable")
        subprocess.run([bash, "-n", str(SCRIPT)], check=True)


if __name__ == "__main__":
    unittest.main()
