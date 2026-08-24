import contextlib
import io
import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from scripts import eval_suite


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENT_HOST = REPO_ROOT / "baselines" / "specedge" / "integration" / "client_host.py"
CONFIG = REPO_ROOT / "configs" / "evaluation" / "qwen3_8b_1.7b_four_method_cpu.json"


class SpecEdgeClientHostTests(unittest.TestCase):
    def test_absolute_client_host_runs_outside_repo_without_pythonpath(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, str(CLIENT_HOST), "--help"],
            cwd=Path(__file__).resolve().anchor,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_node3_specedge_plan_exposes_repo_pythonpath(self):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        execution = config["execution"]

        def fake_layout(current_config, execution=None):
            resolved = execution or current_config["execution"]
            node3_root = (
                PurePosixPath(resolved["node3_repo"])
                / "exp"
                / "comparison"
                / current_config["run_id"]
            )
            node2_root = (
                PurePosixPath(resolved["node2_repo"])
                / "exp"
                / "comparison"
                / current_config["run_id"]
            )
            return {
                "linux_root": node3_root,
                "node3_root": node3_root,
                "node2_root": node2_root,
                "node3_canonical": node3_root / "inputs" / "canonical.jsonl",
                "node2_canonical": node2_root / "inputs" / "canonical.jsonl",
                "node3_specedge_config_linux": node3_root / "specedge" / "specedge.yaml",
                "node2_specedge_config_linux": node2_root / "specedge" / "node2.yaml",
                "commands": node3_root / "commands.txt",
                "status": node3_root / "run_status.jsonl",
            }

        node3_integration = f"{execution['node3_repo']}/baselines/specedge/integration"
        node3_official = f"{execution['node3_repo']}/baselines/specedge/official"
        expected_pythonpath = ":".join(
            (execution["node3_repo"], node3_integration, f"{node3_official}/src")
        )
        output = io.StringIO()
        with patch.object(eval_suite, "load_config", return_value=config), patch.object(
            eval_suite, "output_layout", side_effect=fake_layout
        ), patch.object(
            eval_suite,
            "_read_manifest",
            return_value={"execution": execution, "workload_hash": "test-hash"},
        ), patch.object(eval_suite, "_append_command_record"), patch.object(
            eval_suite, "append_status"
        ), contextlib.redirect_stdout(output):
            self.assertEqual(eval_suite.print_plan(str(CONFIG)), 0)

        plan = output.getvalue()
        self.assertIn(
            f"PYTHONPATH={shlex.quote(expected_pythonpath)}",
            plan,
        )
        self.assertIn(
            f"{execution['node3_repo']}/baselines/specedge/integration/client_host.py",
            plan,
        )


if __name__ == "__main__":
    unittest.main()
