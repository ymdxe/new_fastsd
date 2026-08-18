"""Prepare, validate, normalize, and compare the four-method evaluation suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common_metrics import summarize_requests, write_json
from src.evaluation import (
    load_canonical_jsonl,
    load_evaluation_records,
    workload_fingerprint,
    write_canonical_jsonl,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def output_layout(config: dict[str, Any]) -> dict[str, Any]:
    local_root = REPO_ROOT / "exp" / "comparison" / config["run_id"]
    linux_root = (
        PurePosixPath(config["repo_path_linux"])
        / "exp"
        / "comparison"
        / config["run_id"]
    )
    return {
        "local_root": local_root,
        "linux_root": linux_root,
        "canonical": local_root / "inputs" / "canonical.jsonl",
        "linux_canonical": linux_root / "inputs" / "canonical.jsonl",
        "manifest": local_root / "run_manifest.json",
        "commands": local_root / "commands.txt",
        "specedge_config": local_root / "specedge" / "specedge.yaml",
        "linux_specedge_config": linux_root / "specedge" / "specedge.yaml",
    }


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def render_specedge_config(config: dict[str, Any], workload_hash: str, layout: dict[str, Any]) -> str:
    generation = config["generation"]
    topology = config["topology"]
    draft_devices = topology["specedge_draft_devices"]
    linux_root = layout["linux_root"]
    lines = [
        "version: 1",
        "opt: 2",
        "base:",
        f"  result_path: {_yaml_scalar(linux_root / 'specedge' / 'raw')}",
        f"  exp_name: {_yaml_scalar(config['run_id'])}",
        "  dtype: bf16",
        f"  seed: {int(generation['seed'])}",
        "  ssh_key: ~/.ssh/id_ed25519",
        "  max_len: 4096",
        "server:",
        "  process_name: server",
        f"  target_model: {_yaml_scalar(config['models']['target'])}",
        f"  device: {_yaml_scalar(topology['specedge_target_device'])}",
        f"  temperature: {float(generation['temperature'])}",
        f"  max_batch_size: {len(draft_devices)}",
        f"  num_clients: {len(draft_devices)}",
        "  batch_type: dynamic",
        "  cache_prefill: false",
        "client:",
        f"  host: {_yaml_scalar(topology['specedge_host'])}",
        "  process_name: client",
        f"  draft_model: {_yaml_scalar(config['models']['draft'])}",
        "  dataset: fastsd_external",
        "  reasoning: false",
        "  sample_req_cnt: 1",
        "  req_offset: 0",
        "  max_n_beams: 32",
        f"  max_beam_len: {int(generation['gamma'])}",
        "  max_branch_width: 16",
        "  max_budget: 32",
        "  proactive:",
        "    type: included",
        "    max_n_beams: 32",
        "    max_beam_len: 3",
        "    max_branch_width: 16",
        "    max_budget: 32",
        f"  max_new_tokens: {int(generation['max_new_tokens'])}",
        "  max_request_num: -1",
        "node:",
        "  local:",
    ]
    lines.extend(f"    - device: {_yaml_scalar(device)}" for device in draft_devices)
    lines.extend(
        [
            "integration:",
            "  python: /home/hdd/zhangh/envs/specedge/bin/python",
            "  server_host: 127.0.0.1",
            "  server_port: 18000",
            f"  dataset_file: {_yaml_scalar(layout['linux_canonical'])}",
            f"  completion_dir: {_yaml_scalar(linux_root / 'specedge' / 'requests')}",
            f"  workload_hash: {_yaml_scalar(workload_hash)}",
            f"  arrival_distribution: {_yaml_scalar(config['dataset'].get('arrival_distribution', 'immediate'))}",
            "  startup_delay_s: 15",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_command_record(path: Path, config_path: str | Path, body: str) -> None:
    """Append a timestamped, copyable command block to the run directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# command_record_utc: {timestamp}\n")
        handle.write(f"# config: {config_path}\n")
        handle.write(body.rstrip() + "\n")


def prepare(config_path: str) -> int:
    config = load_config(config_path)
    layout = output_layout(config)
    dataset = config["dataset"]
    generation = config["generation"]
    data_path = Path(dataset["data_path"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    records = load_evaluation_records(
        dataset["name"],
        data_path,
        max_requests=int(dataset.get("max_requests", -1)),
        arrival_distribution=dataset.get("arrival_distribution", "immediate"),
        arrival_rate=float(dataset.get("arrival_rate_rps", 1.0)),
        arrival_seed=int(dataset.get("arrival_seed", 1234)),
    )
    fingerprint = workload_fingerprint(records, generation)
    write_canonical_jsonl(records, layout["canonical"])
    layout["specedge_config"].parent.mkdir(parents=True, exist_ok=True)
    layout["specedge_config"].write_text(
        render_specedge_config(config, fingerprint, layout), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "workload_hash": fingerprint,
        "num_requests": len(records),
        "dataset": dataset,
        "models": config["models"],
        "generation": generation,
        "canonical_dataset": str(layout["linux_canonical"]),
        "methods": ["fastsd", "specedge", "standard_sd", "draft_only"],
    }
    write_json(manifest, layout["manifest"])
    _append_command_record(
        layout["commands"],
        config_path,
        f"cd {REPO_ROOT} && python scripts/eval_suite.py prepare --config {config_path}",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"SpecEdge config: {layout['specedge_config']}")
    print(f"Command record: {layout['commands']}")
    return 0


def _read_manifest(config: dict[str, Any], layout: dict[str, Path]) -> dict[str, Any]:
    if not layout["manifest"].is_file():
        raise FileNotFoundError("run manifest is missing; run prepare first")
    return json.loads(layout["manifest"].read_text(encoding="utf-8"))


def print_plan(config_path: str) -> int:
    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    repo = config["repo_path_linux"]
    run_id = config["run_id"]
    dataset = config["dataset"]["name"]
    generation = config["generation"]
    models = config["models"]
    topology = config["topology"]
    common = (
        f"--dataset {dataset} --dataset_file {layout['linux_canonical']} "
        f"--workload_hash {manifest['workload_hash']} --draft_model {models['draft']} "
        f"--target_model {models['target']} --max_tokens {generation['max_new_tokens']} "
        f"--temp {generation['temperature']} --top_k {generation['top_k']} "
        f"--top_p {generation['top_p']} --gamma {generation['gamma']} --seed {generation['seed']} "
        f"--stop_policy {generation.get('stop_policy', 'eos')}"
    )
    integration = f"{repo}/baselines/specedge/integration"
    official = f"{repo}/baselines/specedge/official"
    specedge_raw = f"{layout['linux_root']}/specedge/raw/{run_id}"
    plan = "\n".join(
        [
            "[两台服务器：先生成完全相同的 workload 文件]",
            f"cd {repo} && /home/hdd/zhangh/envs/fastsd/bin/python scripts/eval_suite.py prepare --config {config_path}",
            "\n[node1：到 node2 的两个本地端口隧道]",
            "ssh -N -L 18000:127.0.0.1:8000 -L 18001:127.0.0.1:8001 node2",
            "\n[node2：FastSD target（FastSD 模式）]",
            f"cd {repo} && FASTSD_TARGET_DEVICE={topology['standard_sd_target_device']} "
            f"/home/hdd/zhangh/envs/fastsd/bin/python cloud/cloud_service.py "
            f"--target_model {models['target']} --draft_model {models['draft']} --dataset {dataset} "
            "--server_sched_mode fastsd",
            "\n[node1：FastSD 方法]",
            f"cd {repo} && SERVER_URL={topology['fastsd_server_url']} bash scripts/run_fastsd_profile.sh "
            f"comparison/{run_id}/fastsd {common} --num_drafts {len(topology['specedge_draft_devices'])} "
            f"--edge_gpus {len(topology['specedge_draft_devices'])} "
            f"--arrival_distribution {config['dataset']['arrival_distribution']} "
            f"--arrival_rate {config['dataset']['arrival_rate_rps']} "
            f"--arrival_seed {config['dataset']['arrival_seed']}",
            "\n[node2：官方 SpecEdge server，使用 canonical dataset hook]",
            f"cd {repo} && FASTSD_EVAL_ROLE=server "
            f"FASTSD_EVAL_DATASET_FILE={layout['linux_canonical']} "
            f"PYTHONPATH={integration}:{official}/src "
            f"/home/hdd/zhangh/envs/specedge/bin/python -O {integration}/server.py "
            f"--config {layout['linux_specedge_config']} --host 127.0.0.1 --port 18000",
            "\n[node1：适配后的 SpecEdge clients]",
            f"cd {repo} && /home/hdd/zhangh/envs/specedge/bin/python {integration}/client_host.py "
            f"--config {layout['linux_specedge_config']}",
            "\n[node2：停止 FastSD target 后，以 vanilla 调度重启 target]",
            f"cd {repo} && FASTSD_TARGET_DEVICE={topology['standard_sd_target_device']} "
            f"/home/hdd/zhangh/envs/fastsd/bin/python cloud/cloud_service.py "
            f"--target_model {models['target']} --draft_model {models['draft']} --dataset {dataset} "
            "--server_sched_mode vanilla",
            "\n[node1：标准投机解码（同样的 A5000 draft + 网络 + A6000 target）]",
            f"cd {repo} && SERVER_URL={topology['fastsd_server_url']} bash scripts/run_vanilla_profile.sh "
            f"vanilla comparison/{run_id}/standard_sd {common} "
            f"--num_drafts {len(topology['specedge_draft_devices'])} "
            f"--edge_gpus {len(topology['specedge_draft_devices'])} "
            f"--arrival_distribution {config['dataset']['arrival_distribution']} "
            f"--arrival_rate {config['dataset']['arrival_rate_rps']} "
            f"--arrival_seed {config['dataset']['arrival_seed']}",
            "\n[node1：仅 Draft 模型，同数量 A5000 worker]",
            f"cd {repo} && /home/hdd/zhangh/envs/fastsd/bin/python benchmark/eval_draft_pool.py "
            f"--config {config_path}",
            f"\n[SpecEdge raw result expected at {specedge_raw}]",
        ]
    )
    print(plan)
    _append_command_record(layout["commands"], config_path, plan)
    print(f"\nCommand record: {layout['commands']}")
    return 0


def _read_jsonl_files(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    return records


def normalize_fastsd(input_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict], float | None]:
    raw = _read_jsonl_files(sorted(input_dir.glob("edge_metrics_proc*.jsonl")))
    records = [
        {
            "sample_id": item.get("task_id"),
            "generated_tokens": item.get("generated_tokens", 0),
            "ttft_ms": item.get("ttft_ms"),
            "tpot_ms": item.get("tpot_ms"),
            "e2e_ms": item.get("request_e2e_ms"),
            "scheduled_arrival_s": item.get("scheduled_arrival_s"),
            "actual_arrival_s": item.get("actual_arrival_s"),
            "arrival_lag_ms": item.get("arrival_lag_ms"),
            "completion_s": item.get("completion_s"),
            "accepted_tokens": item.get("accepted_total", 0),
            "drafted_tokens": item.get("drafted_total", 0),
            "mean_accepted_tokens_per_verify": item.get(
                "mean_accepted_tokens_per_verify"
            ),
            "output_text": item.get("output_text"),
            "reference": item.get("reference"),
        }
        for item in raw
    ]
    edge_summary = input_dir / "edge_metrics_summary.json"
    wallclock = None
    if edge_summary.is_file():
        summary_payload = json.loads(edge_summary.read_text(encoding="utf-8"))
        if summary_payload.get("wallclock_s") is not None:
            wallclock = float(summary_payload["wallclock_s"])
    return records, wallclock


def normalize_specedge(input_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict], float | None]:
    cycle_files = sorted(input_dir.glob("client_*.jsonl"))
    cycles = _read_jsonl_files(cycle_files)
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for item in cycles:
        grouped[(int(item["client_idx"]), int(item["req_idx"]))].append(item)

    completion_root = input_dir.parents[1] / "requests"
    completions = _read_jsonl_files(sorted(completion_root.glob("client_*_requests.jsonl")))
    completion_by_global = {int(item["global_index"]): item for item in completions}
    records = []
    for (_client_idx, global_index), items in grouped.items():
        ordered = sorted(items, key=lambda item: int(item["step_idx"]))
        tokens = [int(item.get("num_accepted_tokens", 0)) for item in ordered]
        completion = completion_by_global.get(global_index, {})
        generated = int(completion.get("generated_tokens", sum(tokens)))
        records.append(
            {
                "sample_id": completion.get("sample_id", str(global_index)),
                "generated_tokens": generated,
                "ttft_ms": completion.get("ttft_ms"),
                "tpot_ms": completion.get("tpot_ms"),
                "e2e_ms": completion.get("request_e2e_ms"),
                "scheduled_arrival_s": completion.get("scheduled_arrival_s"),
                "actual_arrival_s": completion.get("actual_arrival_s"),
                "arrival_lag_ms": completion.get("arrival_lag_ms"),
                "completion_s": completion.get("completion_s"),
                "mean_accepted_tokens_per_verify": statistics.mean(tokens[1:]) if len(tokens) > 1 else None,
                "output_text": completion.get("output_text"),
                "reference": completion.get("reference"),
            }
        )
    return records, None


def normalize(config_path: str, method: str, input_path: str) -> int:
    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    source = Path(input_path)
    if method == "fastsd":
        records, wallclock = normalize_fastsd(source, manifest)
    elif method == "specedge":
        records, wallclock = normalize_specedge(source, manifest)
    elif method == "standard_sd":
        records, wallclock = normalize_fastsd(source, manifest)
    elif method == "draft_only":
        records = _read_jsonl_files([source / "requests.jsonl"])
        wallclock = None
    else:
        raise ValueError(f"Unsupported method: {method}")
    output_dir = layout["local_root"] / "normalized" / method
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if config["dataset"]["name"] == "humaneval":
        humaneval_path = output_dir / "humaneval_samples.jsonl"
        with humaneval_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {
                            "task_id": record["sample_id"],
                            "completion": record.get("output_text", ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    summary = summarize_requests(
        records,
        method=method,
        dataset=config["dataset"]["name"],
        workload_hash=manifest["workload_hash"],
        run_id=config["run_id"],
        wallclock_s=wallclock,
        extra={"draft_model": config["models"]["draft"], "target_model": config["models"]["target"]},
    )
    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def compare(summary_paths: list[str], output: str | None) -> int:
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in summary_paths]
    hashes = {summary["workload_hash"] for summary in summaries}
    if len(hashes) != 1:
        raise ValueError(f"workload hashes differ: {sorted(hashes)}")
    columns = [
        "method", "num_requests", "throughput_tok_s", "ttft_avg_ms", "ttft_p95_ms",
        "scheduled_ttft_p95_ms",
        "tpot_avg_ms", "tpot_p95_ms", "e2e_avg_ms", "accept_rate",
        "mean_accepted_tokens_per_verify", "quality_exact_match",
    ]
    rows = []
    for item in summaries:
        rows.append({
            "method": item["method"],
            "num_requests": item["num_requests"],
            "throughput_tok_s": item["throughput_tok_s"],
            "ttft_avg_ms": item["ttft_ms"]["avg"],
            "ttft_p95_ms": item["ttft_ms"]["p95"],
            "scheduled_ttft_p95_ms": item["scheduled_ttft_ms"]["p95"],
            "tpot_avg_ms": item["tpot_ms"]["avg"],
            "tpot_p95_ms": item["tpot_ms"]["p95"],
            "e2e_avg_ms": item["e2e_ms"]["avg"],
            "accept_rate": item["accept_rate"],
            "mean_accepted_tokens_per_verify": item["mean_accepted_tokens_per_verify"],
            "quality_exact_match": item["quality_exact_match"],
        })
    print("\t".join(columns))
    for row in rows:
        print("\t".join("" if row[col] is None else str(row[col]) for col in columns))
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_models(config_path: str) -> int:
    config = load_config(config_path)
    details = {}
    failed = False
    for role in ("draft", "target"):
        path = Path(config["models"][role])
        if not path.is_dir():
            print(f"[FAIL] {role} model path missing: {path}")
            failed = True
            continue
        model_config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        tokenizer_file = path / "tokenizer.json"
        if not tokenizer_file.is_file():
            print(f"[FAIL] {role} tokenizer.json missing: {tokenizer_file}")
            failed = True
        details[role] = {
            "path": str(path),
            "model_type": model_config.get("model_type"),
            "vocab_size": model_config.get("vocab_size"),
            "tokenizer_sha256": _sha256(tokenizer_file) if tokenizer_file.is_file() else None,
        }
        if details[role]["model_type"] not in {"qwen3", "llama"}:
            print(f"[FAIL] official SpecEdge does not support model_type={details[role]['model_type']}")
            failed = True
    if len(details) == 2:
        for key in ("vocab_size", "tokenizer_sha256"):
            if details["draft"][key] != details["target"][key]:
                print(f"[FAIL] draft/target {key} differs")
                failed = True
        if not failed:
            print("[PASS] model architecture and tokenizer fingerprints are compatible")
    print(json.dumps(details, indent=2, ensure_ascii=False))
    return int(failed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "plan", "validate-models"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    normalizer = subparsers.add_parser("normalize")
    normalizer.add_argument("--config", required=True)
    normalizer.add_argument("--method", required=True, choices=["fastsd", "specedge", "standard_sd", "draft_only"])
    normalizer.add_argument("--input", required=True)
    comparator = subparsers.add_parser("compare")
    comparator.add_argument("summaries", nargs="+")
    comparator.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args.config)
    if args.command == "plan":
        return print_plan(args.config)
    if args.command == "normalize":
        return normalize(args.config, args.method, args.input)
    if args.command == "compare":
        return compare(args.summaries, args.output)
    if args.command == "validate-models":
        return validate_models(args.config)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
