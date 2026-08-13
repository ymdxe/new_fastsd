import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from scripts.eval_suite import (
    normalize_fastsd,
    normalize_specedge,
    output_layout,
    render_specedge_config,
)
from src.common_metrics import summarize_requests
from src.evaluation import (
    load_canonical_jsonl,
    load_evaluation_records,
    workload_fingerprint,
    write_canonical_jsonl,
)


class EvaluationDatasetTests(unittest.TestCase):
    def test_all_repository_dataset_schemas_are_adapted(self):
        fixtures = {
            "humaneval": {"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "assert f()"},
            "gsm8k": {"question": "1+1?", "answer": "#### 2"},
            "mgsm": {"question_id": "m1", "question": "1+1?", "answer": "2"},
            "mt_bench": {"question_id": 7, "category": "math", "turns": ["hello", "again"]},
        }
        names = {
            "humaneval": "humaneval.jsonl",
            "gsm8k": "gsm8k.jsonl",
            "mgsm": "mgsm.jsonl",
            "mt_bench": "mt_bench.jsonl",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for dataset, fixture in fixtures.items():
                (root / names[dataset]).write_text(json.dumps(fixture) + "\n", encoding="utf-8")
                records = load_evaluation_records(dataset, root)
                self.assertEqual(len(records), 1)
                self.assertTrue(records[0].prompt)
                self.assertEqual(records[0].dataset, dataset)

    def test_poisson_manifest_is_deterministic_and_round_trips(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "gsm8k.jsonl"
            source.write_text(
                "\n".join(json.dumps({"question": str(i), "answer": str(i)}) for i in range(3)),
                encoding="utf-8",
            )
            left = load_evaluation_records("gsm8k", source, arrival_distribution="poisson", arrival_seed=9)
            right = load_evaluation_records("gsm8k", source, arrival_distribution="poisson", arrival_seed=9)
            self.assertEqual(left, right)
            self.assertEqual(workload_fingerprint(left, {"seed": 1}), workload_fingerprint(right, {"seed": 1}))
            canonical = Path(temp_dir) / "canonical.jsonl"
            write_canonical_jsonl(left, canonical)
            loaded = load_canonical_jsonl(canonical)
            self.assertEqual([item["global_index"] for item in loaded], [0, 1, 2])
            self.assertTrue(all(loaded[i]["scheduled_arrival_s"] < loaded[i + 1]["scheduled_arrival_s"] for i in range(2)))


class CommonMetricTests(unittest.TestCase):
    def test_summary_reports_service_and_queue_inclusive_ttft(self):
        summary = summarize_requests(
            [
                {
                    "generated_tokens": 3,
                    "ttft_ms": 10,
                    "arrival_lag_ms": 5,
                    "tpot_ms": 2,
                    "e2e_ms": 14,
                    "actual_arrival_s": 1,
                    "completion_s": 1.014,
                    "output_text": "2",
                    "reference": "#### 2",
                }
            ],
            method="draft_only",
            dataset="gsm8k",
            workload_hash="abc",
            run_id="run",
        )
        self.assertEqual(summary["ttft_ms"]["avg"], 10)
        self.assertEqual(summary["scheduled_ttft_ms"]["avg"], 15)
        self.assertEqual(summary["quality_exact_match"], 1.0)


class DraftOnlyPromptTests(unittest.TestCase):
    def test_mt_bench_uses_qwen_chat_template_without_thinking(self):
        source = (Path(__file__).parents[1] / "benchmark" / "eval_draft_pool.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn('arrival_distribution == "poisson"', source)
        self.assertIn("actual_arrival = 0.0", source)


class SpecEdgeAdapterTests(unittest.TestCase):
    def _config(self):
        return {
            "run_id": "run",
            "repo_path_linux": "/srv/new_fastsd",
            "dataset": {"arrival_distribution": "immediate"},
            "models": {"draft": "/models/Qwen3-0.6B", "target": "/models/Qwen3-8B"},
            "generation": {"seed": 42, "temperature": 0, "gamma": 4, "max_new_tokens": 8},
            "topology": {
                "specedge_host": "127.0.0.1:18000",
                "specedge_target_device": "cuda:0",
                "specedge_draft_devices": ["cuda:0", "cuda:1"],
            },
        }

    def test_linux_paths_remain_posix_on_windows(self):
        layout = output_layout(self._config())
        self.assertIsInstance(layout["linux_root"], PurePosixPath)
        self.assertEqual(str(layout["linux_canonical"]), "/srv/new_fastsd/exp/comparison/run/inputs/canonical.jsonl")

    def test_rendered_yaml_has_two_client_devices(self):
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML is unavailable")
        layout = output_layout(self._config())
        rendered = yaml.safe_load(render_specedge_config(self._config(), "hash", layout))
        self.assertEqual(rendered["server"]["max_batch_size"], 2)
        self.assertEqual(rendered["node"]["local"], [{"device": "cuda:0"}, {"device": "cuda:1"}])
        self.assertEqual(rendered["base"]["dtype"], "bf16")
        self.assertEqual(rendered["integration"]["arrival_distribution"], "immediate")
        self.assertEqual(rendered["integration"]["server_port"], 18000)
        self.assertEqual(
            rendered["integration"]["python"],
            "/home/hdd/zhangh/envs/specedge/bin/python",
        )

    def test_specedge_mt_bench_adapter_uses_chat_template_and_immediate_arrivals(self):
        source = (
            Path(__file__).parents[1]
            / "baselines"
            / "specedge"
            / "integration"
            / "client.py"
        ).read_text(encoding="utf-8")
        self.assertIn("tokenizer.apply_chat_template", source)
        self.assertIn("enable_thinking=False", source)
        self.assertIn("actual_arrival = 0.0", source)
        server_source = (
            Path(__file__).parents[1]
            / "baselines"
            / "specedge"
            / "integration"
            / "server.py"
        ).read_text(encoding="utf-8")
        self.assertIn("official_server.SpecExecBatchServer", server_source)
        self.assertIn("server.add_insecure_port", server_source)

    def test_specedge_normalizer_uses_precise_client_measurements(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir) / "specedge" / "raw" / "run"
            requests = Path(temp_dir) / "specedge" / "requests"
            raw.mkdir(parents=True)
            requests.mkdir(parents=True)
            cycles = [
                {"client_idx": 0, "req_idx": 2, "step_idx": 0, "draft": {"end_to_end": 3}, "target": {"end_to_end": 7}, "num_accepted_tokens": 1},
                {"client_idx": 0, "req_idx": 2, "step_idx": 1, "draft": {"end_to_end": 4}, "target": {"end_to_end": 8}, "num_accepted_tokens": 3},
            ]
            (raw / "client_0.jsonl").write_text("\n".join(json.dumps(item) for item in cycles), encoding="utf-8")
            completion = {
                "global_index": 2,
                "sample_id": "HumanEval/2",
                "generated_tokens": 4,
                "ttft_ms": 13,
                "tpot_ms": 5,
                "request_e2e_ms": 28,
                "arrival_lag_ms": 2,
            }
            (requests / "client_0_requests.jsonl").write_text(json.dumps(completion), encoding="utf-8")
            normalized, wallclock = normalize_specedge(raw, {"workload_hash": "hash"})
            self.assertIsNone(wallclock)
            self.assertEqual(normalized[0]["ttft_ms"], 13)
            self.assertEqual(normalized[0]["tpot_ms"], 5)
            self.assertEqual(normalized[0]["mean_accepted_tokens_per_verify"], 3)

    def test_fastsd_normalizer_uses_run_wallclock_for_closed_loop_workers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw = Path(temp_dir)
            request = {
                "task_id": "81",
                "generated_tokens": 8,
                "ttft_ms": 10,
                "tpot_ms": 2,
                "request_e2e_ms": 24,
                "completion_s": 0.024,
                "actual_arrival_s": 0.0,
                "accepted_total": 3,
                "drafted_total": 8,
                "mean_accepted_tokens_per_verify": 1.5,
            }
            (raw / "edge_metrics_proc0.jsonl").write_text(
                json.dumps(request) + "\n", encoding="utf-8"
            )
            (raw / "edge_metrics_summary.json").write_text(
                json.dumps({"wallclock_s": 30.0}), encoding="utf-8"
            )

            normalized, wallclock = normalize_fastsd(raw, {"workload_hash": "hash"})
            self.assertEqual(wallclock, 30.0)
            self.assertEqual(normalized[0]["mean_accepted_tokens_per_verify"], 1.5)


if __name__ == "__main__":
    unittest.main()
