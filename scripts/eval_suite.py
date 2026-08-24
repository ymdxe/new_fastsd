"""Prepare, validate, normalize, and compare the four-method evaluation suite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import statistics
import sys
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FORMAL_CPUSET = "56-71,80-95"
FORMAL_CPU_PREFIX = "nice -n 5 numactl --physcpubind=56-71,80-95 --interleave=2,3"

from src.common_metrics import summarize_requests
from src.evaluation import (
    DATASET_FILES,
    load_canonical_jsonl,
    load_evaluation_records,
    workload_fingerprint,
    write_canonical_jsonl,
)
from src.parity import FOUR_METHODS, ORACLE_METHOD, build_token_parity_rows, write_token_parity_report
from src.run_artifacts import (
    append_command,
    append_status,
    copy_file_once,
    git_sha,
    require_unique_sample_ids,
    write_json_once,
    write_text_once,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def shell_prefix(value: Any) -> tuple[str, str]:
    """Return a safe shell rendering and the raw prefix for audit records."""

    if value is None:
        return "", ""
    if isinstance(value, list):
        if not all(isinstance(item, str) and item for item in value):
            raise ValueError("topology.cpu_prefix list entries must be non-empty strings")
        raw = " ".join(value)
        return shlex.join(value) + " ", raw
    if not isinstance(value, str):
        raise ValueError("topology.cpu_prefix must be a string or argv list")
    raw = value
    if not raw.strip():
        return "", raw
    try:
        argv = shlex.split(raw, posix=True)
    except ValueError as exc:
        raise ValueError(f"invalid topology.cpu_prefix: {exc}") from exc
    if not argv:
        return "", raw
    return shlex.join(argv) + " ", raw


def prefixed_command(
    prefix: str,
    command: str,
    *,
    environment: dict[str, Any] | None = None,
) -> str:
    """Place env assignments after taskset/numactl and before the command."""

    env_text = ""
    if environment:
        assignments = " ".join(
            f"{key}={shlex.quote(str(value))}" for key, value in environment.items()
        )
        env_text = f"env {assignments} "
    return f"{prefix}{env_text}{command}"


def validate_experiment_config(config: dict[str, Any]) -> list[str]:
    """Validate the frozen CPU experiment tracks.

    Older comparison configurations predate the explicit ``track`` field and
    remain renderable for backwards compatibility.  Once a configuration opts
    into a named track, however, silently mixing latency and throughput
    settings would invalidate the comparison, so fail early with a useful
    error instead of producing a misleading run directory.
    """

    generation = config.get("generation", {})
    topology = config.get("topology", {})
    dataset = config.get("dataset", {})
    max_new_tokens = int(generation.get("max_new_tokens", 0))
    if max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    if max_new_tokens <= 16:
        scope = config.get("evaluation_scope", topology.get("evaluation_scope"))
        if scope not in {None, "communication_smoke"}:
            raise ValueError(
                "max_new_tokens <= 16 is communication_smoke only; "
                f"got evaluation_scope={scope!r}"
            )

    track = topology.get("track")
    if track is None:
        return []
    if track not in {"latency", "throughput"}:
        raise ValueError("topology.track must be 'latency' or 'throughput'")

    arrival = dataset.get("arrival_distribution", "immediate")
    concurrency = int(topology.get("concurrency", 1))
    workers = int(topology.get("draft_workers", 1))
    threads = int(topology.get("draft_threads", 1))
    warmups = int(topology.get("warmup_requests", 0))
    if warmups < 0:
        raise ValueError("topology.warmup_requests must be non-negative")

    if track == "latency":
        expected = {
            "dataset.arrival_distribution": (arrival, "immediate"),
            "topology.concurrency": (concurrency, 1),
            "topology.draft_workers": (workers, 1),
            "topology.draft_threads": (threads, 32),
            "topology.warmup_requests": (warmups, 10),
        }
    else:
        expected = {
            "dataset.arrival_distribution": (arrival, "poisson"),
            "topology.concurrency": (concurrency, 4),
            "topology.draft_workers": (workers, 4),
            "topology.draft_threads": (threads, 8),
        }
    mismatches = [
        f"{name}={actual!r}, expected {expected_value!r} for {track} track"
        for name, (actual, expected_value) in expected.items()
        if actual != expected_value
    ]
    if mismatches:
        raise ValueError("invalid frozen track configuration: " + "; ".join(mismatches))

    draft_devices = list(topology.get("specedge_draft_devices", []))
    if draft_devices and not all(
        str(device).split(":", 1)[0].lower() == "cpu" for device in draft_devices
    ):
        raise ValueError("the CPU comparison tracks require CPU SpecEdge draft devices")
    draft_only_devices = list(topology.get("draft_only_devices", draft_devices))
    if draft_only_devices and not all(
        str(device).split(":", 1)[0].lower() == "cpu" for device in draft_only_devices
    ):
        raise ValueError("the CPU comparison tracks require CPU draft-only devices")
    _rendered_prefix, raw_prefix = shell_prefix(topology.get("cpu_prefix", ""))
    if topology.get("shared_load", True) is not True:
        raise ValueError("formal CPU comparison requires topology.shared_load=true")
    physical_cpu_count = int(topology.get("physical_cpu_count", 32))
    numa_cores_per_node = int(topology.get("numa_cores_per_node", 16))
    if physical_cpu_count != 32:
        raise ValueError("formal CPU comparison fixes physical_cpu_count=32")
    if numa_cores_per_node != 16:
        raise ValueError("formal CPU comparison fixes numa_cores_per_node=16")
    numa_node_ids = list(topology.get("numa_node_ids", [2, 3]))
    if numa_node_ids != [2, 3]:
        raise ValueError("formal CPU comparison fixes NUMA nodes [2, 3]")
    fixed_cpuset = topology.get("fixed_cpuset", "")
    if not isinstance(fixed_cpuset, str):
        raise ValueError("topology.fixed_cpuset must be a string")
    if fixed_cpuset != FORMAL_CPUSET:
        raise ValueError(
            f"formal CPU comparison requires fixed_cpuset={FORMAL_CPUSET}; "
            "older 8-15/32-39 candidates are forbidden"
        )
    if shlex.split(raw_prefix) != shlex.split(FORMAL_CPU_PREFIX):
        raise ValueError(
            "formal CPU comparison requires cpu_prefix='nice -n 5 numactl "
            "--physcpubind=56-71,80-95 --interleave=2,3'"
        )
    method_order = list(topology.get("method_order", FOUR_METHODS))
    if method_order != list(FOUR_METHODS) and sorted(method_order) != sorted(FOUR_METHODS):
        raise ValueError(
            "topology.method_order must contain each four-method label exactly once"
        )
    repeat_index = int(topology.get("repeat_index", 0))
    if repeat_index < 0:
        raise ValueError("topology.repeat_index must be non-negative")

    notes = [
        "max_new_tokens <= 16 is communication_smoke only"
        if max_new_tokens <= 16
        else "max_new_tokens > 16 is eligible for quality/performance reporting"
    ]
    notes.append(
        "shared-load fair comparison: fixed cpuset 56-71,80-95; "
        "NUMA nodes 2/3 with 16 physical cores each"
    )
    return notes


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
        "raw_input": local_root / "inputs" / "raw_input.jsonl",
        "linux_canonical": linux_root / "inputs" / "canonical.jsonl",
        "manifest": local_root / "run_manifest.json",
        "commands": local_root / "commands.txt",
        "status": local_root / "run_status.jsonl",
        "cpu_preflight": local_root / "cpu_preflight.json",
        "cpu_postflight": local_root / "cpu_postflight.json",
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
    draft_is_cpu = bool(draft_devices) and all(
        str(device).split(":", 1)[0].lower() == "cpu" for device in draft_devices
    )
    client_dtype = "fp32" if draft_is_cpu else "bf16"
    client_engine = "cpu_adapter" if draft_is_cpu else "official_graph_engine"
    client_threads = int(topology.get("draft_threads", 1))
    linux_root = layout["linux_root"]
    integration_python = config.get("execution", {}).get("specedge_python", "python")
    specedge_host = topology.get("specedge_host", "127.0.0.1:18000")
    specedge_host_name = str(specedge_host).rsplit(":", 1)[0]
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
        f"  host: {_yaml_scalar(specedge_host)}",
        "  process_name: client",
        f"  draft_model: {_yaml_scalar(config['models']['draft'])}",
        "  dataset: fastsd_external",
        f"  dtype: {client_dtype}",
        f"  engine: {client_engine}",
        f"  threads: {client_threads}",
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
            f"  python: {_yaml_scalar(integration_python)}",
            f"  server_host: {_yaml_scalar(specedge_host_name)}",
            f"  server_port: {int(topology.get('specedge_port', 18000))}",
            f"  dataset_file: {_yaml_scalar(layout['linux_canonical'])}",
            f"  completion_dir: {_yaml_scalar(linux_root / 'specedge' / 'requests')}",
            f"  workload_hash: {_yaml_scalar(workload_hash)}",
            f"  method: {_yaml_scalar('specedge_cpu_adapted' if draft_is_cpu else 'specedge')}",
            f"  commands_path: {_yaml_scalar(layout['linux_root'] / 'commands.txt')}",
            f"  status_path: {_yaml_scalar(layout['linux_root'] / 'run_status.jsonl')}",
            f"  arrival_distribution: {_yaml_scalar(config['dataset'].get('arrival_distribution', 'immediate'))}",
            f"  warmup_requests: {int(topology.get('warmup_requests', 0))}",
            "  startup_delay_s: 15",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_command_record(path: Path, config_path: str | Path, body: str) -> None:
    """Append a timestamped, copyable command block to the run directory."""
    append_command(path, body, config=config_path)


def prepare(config_path: str) -> int:
    config = load_config(config_path)
    validation_notes = validate_experiment_config(config)
    layout = output_layout(config)
    dataset = config["dataset"]
    generation = config["generation"]
    data_path = Path(dataset["data_path"])
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path
    raw_source = data_path / DATASET_FILES[dataset["name"]] if data_path.is_dir() else data_path
    records = load_evaluation_records(
        dataset["name"],
        data_path,
        max_requests=int(dataset.get("max_requests", -1)),
        arrival_distribution=dataset.get("arrival_distribution", "immediate"),
        arrival_rate=float(dataset.get("arrival_rate_rps", 1.0)),
        arrival_seed=int(dataset.get("arrival_seed", 1234)),
    )
    fingerprint = workload_fingerprint(records, generation)
    require_unique_sample_ids(
        {
            "sample_id": record.sample_id
        }
        for record in records
    )
    canonical_payload = "".join(
        json.dumps(
            {
                **record.__dict__,
                "global_index": index,
            },
            ensure_ascii=False,
        )
        + "\n"
        for index, record in enumerate(records)
    )
    write_text_once(layout["canonical"], canonical_payload)
    copy_file_once(raw_source, layout["raw_input"])
    layout["specedge_config"].parent.mkdir(parents=True, exist_ok=True)
    write_text_once(
        layout["specedge_config"], render_specedge_config(config, fingerprint, layout)
    )
    max_new_tokens = int(generation.get("max_new_tokens", 0))
    evaluation_scope = "communication_smoke" if max_new_tokens <= 16 else "quality"
    draft_devices = list(config["topology"].get("specedge_draft_devices", []))
    methods = list(FOUR_METHODS)
    git_info = {"fastsd": git_sha(REPO_ROOT)}
    official_root = REPO_ROOT / "baselines" / "specedge" / "official"
    git_info["specedge_official"] = git_sha(official_root)
    manifest = {
        "schema_version": 2,
        "run_id": config["run_id"],
        "workload_hash": fingerprint,
        "num_requests": len(records),
        "dataset": dataset,
        "models": config["models"],
        "generation": generation,
        "evaluation_scope": evaluation_scope,
        "mt_bench_turn_policy": (
            "first_turn_only" if dataset["name"] == "mt_bench" else "not_applicable"
        ),
        "canonical_dataset": str(layout["linux_canonical"]),
        "raw_input": str(layout["raw_input"]),
        "git_sha": git_info,
        "methods": methods,
        "oracle_method": ORACLE_METHOD,
        "topology": {
            "draft_devices": draft_devices,
            "draft_threads": config["topology"].get("draft_threads"),
            "thread_environment": {
                "OMP_NUM_THREADS": config["topology"].get("draft_threads"),
                "MKL_NUM_THREADS": config["topology"].get("draft_threads"),
                "OPENBLAS_NUM_THREADS": config["topology"].get("draft_threads"),
                "TORCH_NUM_THREADS": config["topology"].get("draft_threads"),
            },
            "track": config["topology"].get("track", "latency"),
            "concurrency": config["topology"].get("concurrency"),
            "warmup_requests": config["topology"].get("warmup_requests", 0),
            "cpu_prefix": config["topology"].get("cpu_prefix", ""),
            "shared_load": bool(config["topology"].get("shared_load", True)),
            "fixed_cpuset": config["topology"].get("fixed_cpuset", ""),
            "physical_cpu_count": config["topology"].get("physical_cpu_count"),
            "numa_cores_per_node": config["topology"].get("numa_cores_per_node"),
            "method_order": config["topology"].get("method_order", list(FOUR_METHODS)),
            "repeat_index": config["topology"].get("repeat_index", 0),
            "cpu_preflight": str(layout["linux_root"] / "cpu_preflight.json"),
            "cpu_postflight": str(layout["linux_root"] / "cpu_postflight.json"),
            "target_host": config["topology"].get("target_host", "127.0.0.1"),
            "target_port": config["topology"].get("target_port", 18001),
            "target_bind_host": config["topology"].get("target_bind_host", "127.0.0.1"),
            "specedge_host": config["topology"].get("specedge_host", "127.0.0.1:18000"),
            "specedge_bind_host": config["topology"].get("specedge_bind_host", "127.0.0.1"),
            "specedge_port": config["topology"].get("specedge_port", 18000),
        },
        "validation_notes": validation_notes,
    }
    write_json_once(layout["manifest"], manifest)
    _append_command_record(
        layout["commands"],
        config_path,
        f"cd {REPO_ROOT} && python scripts/eval_suite.py prepare --config {config_path}",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"SpecEdge config: {layout['specedge_config']}")
    print(f"Command record: {layout['commands']}")
    append_status(
        layout["status"],
        method="suite",
        phase="prepare",
        exit_code=0,
        command=f"python scripts/eval_suite.py prepare --config {config_path}",
    )
    return 0


def _read_manifest(config: dict[str, Any], layout: dict[str, Path]) -> dict[str, Any]:
    if not layout["manifest"].is_file():
        raise FileNotFoundError("run manifest is missing; run prepare first")
    return json.loads(layout["manifest"].read_text(encoding="utf-8"))


def print_plan(
    config_path: str,
    *,
    python_bin: str | None = None,
    target_host: str | None = None,
    target_port: int | None = None,
    cpu_prefix: str | None = None,
) -> int:
    config = load_config(config_path)
    validate_experiment_config(config)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    repo = config["repo_path_linux"]
    run_id = config["run_id"]
    dataset = config["dataset"]["name"]
    generation = config["generation"]
    models = config["models"]
    topology = config["topology"]
    execution = config.get("execution", {})
    python_bin = python_bin or execution.get("python", "python")
    target_python = execution.get("target_python", python_bin)
    specedge_python = execution.get("specedge_python", python_bin)
    draft_devices = list(topology.get("specedge_draft_devices", ["cpu"]))
    worker_count = int(topology.get("draft_workers", len(draft_devices)))
    draft_threads = int(topology.get("draft_threads", 32 if worker_count == 1 else 8))
    physical_cpu_count = int(topology.get("physical_cpu_count", 32))
    target_host = target_host or topology.get("target_host", "127.0.0.1")
    target_port = int(target_port or topology.get("target_port", 18001))
    warmup_requests = int(topology.get("warmup_requests", 0))
    raw_cpu_prefix = cpu_prefix if cpu_prefix is not None else topology.get("cpu_prefix", "")
    command_prefix, raw_cpu_prefix = shell_prefix(raw_cpu_prefix)
    cpu_environment = {
        "OMP_NUM_THREADS": draft_threads,
        "MKL_NUM_THREADS": draft_threads,
        "OPENBLAS_NUM_THREADS": draft_threads,
        "TORCH_NUM_THREADS": draft_threads,
    }
    target_url = f"http://{target_host}:{target_port}"
    target_bind_host = str(topology.get("target_bind_host", "127.0.0.1"))
    specedge_host = topology.get(
        "specedge_host",
        f"{target_host}:{int(topology.get('specedge_port', 18000))}",
    )
    specedge_bind_host = str(topology.get("specedge_bind_host", "127.0.0.1"))
    specedge_port = int(topology.get("specedge_port", 18000))
    draft_device = str(draft_devices[0] if draft_devices else "cpu")
    cpu_draft = draft_device.split(":", 1)[0].lower() == "cpu"
    common = (
        f"--dataset {dataset} --dataset_file {layout['linux_canonical']} "
        f"--workload_hash {manifest['workload_hash']} --draft_model {models['draft']} "
        f"--target_model {models['target']} --max_tokens {generation['max_new_tokens']} "
        f"--temp {generation['temperature']} --top_k {generation['top_k']} "
        f"--top_p {generation['top_p']} --gamma {generation['gamma']} --seed {generation['seed']} "
        f"--stop_policy {generation.get('stop_policy', 'eos')} "
        f"--warmup_requests {warmup_requests}"
    )
    integration = f"{repo}/baselines/specedge/integration"
    official = f"{repo}/baselines/specedge/official"
    specedge_raw = f"{layout['linux_root']}/specedge/raw/{run_id}"
    plan = "\n".join(
        [
            "[两台服务器：先生成完全相同的 workload 文件；不下载数据/模型]",
            f"# cpu_prefix_raw={json.dumps(raw_cpu_prefix, ensure_ascii=False)}",
            f"# shared_load={str(bool(topology.get('shared_load', True))).lower()} "
            f"fixed_cpuset={json.dumps(str(topology.get('fixed_cpuset', '')), ensure_ascii=False)} "
            f"method_order={json.dumps(topology.get('method_order', list(FOUR_METHODS)), ensure_ascii=False)} "
            f"repeat_index={int(topology.get('repeat_index', 0))}",
            f"cd {repo} && {shlex.quote(python_bin)} scripts/eval_suite.py prepare --config {config_path}",
            "\n[node3：CPU 绑定/背景负载预检（不假设 32 核独占；必须先于正式方法）]",
            f"cd {repo} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(python_bin)} scripts/preflight_cpu.py "
                f"--threads {draft_threads} --expected-cpu-count {physical_cpu_count} "
                f"--expected-cpuset {shlex.quote(str(topology.get('fixed_cpuset', '')))} --phase pre "
                f"--output {layout['linux_root']}/cpu_preflight.json "
                f"--commands-path {layout['linux_root']}/commands.txt",
                environment=cpu_environment,
            ),
            "\n[node2：FastSD target（target_host/port 与 bind host 可配置；物理 GPU 由外部 CUDA_VISIBLE_DEVICES 选择）]",
            f"cd {repo} && CLOUD_SERVICE_HOST={shlex.quote(target_bind_host)} CLOUD_SERVICE_PORT={target_port} "
            f"FASTSD_TARGET_DEVICE={topology['standard_sd_target_device']} "
            f"{shlex.quote(target_python)} cloud/cloud_service.py "
            f"--target_model {models['target']} --draft_model {models['draft']} --dataset {dataset} "
            "--server_sched_mode fastsd",
            "\n[node3：FastSD stateful EdgeClient（/session/init + /prefill + /verify + rollback）]",
            f"cd {repo} && "
            + prefixed_command(
                command_prefix,
                "bash scripts/run_fastsd_profile.sh "
                f"comparison/{run_id}/fastsd {common} --num_drafts {worker_count} "
                f"--edge_gpus 1 {'--edge_use_cpu' if cpu_draft else '--no-edge_use_cpu'} --edge_threads {draft_threads} "
                f"--arrival_distribution {config['dataset']['arrival_distribution']} "
                f"--arrival_rate {config['dataset']['arrival_rate_rps']} "
                f"--arrival_seed {config['dataset']['arrival_seed']}",
                environment={
                    **cpu_environment,
                    "PYTHON_BIN": python_bin,
                    "SERVER_URL": target_url,
                },
            ),
            "\n[node2：SpecEdge target server（official target/tree verification core；bind host 可配置）]",
            f"cd {repo} && FASTSD_EVAL_ROLE=server "
            f"FASTSD_EVAL_DATASET_FILE={layout['linux_canonical']} "
            f"PYTHONPATH={integration}:{official}/src "
            f"{shlex.quote(specedge_python)} -O {integration}/server.py "
            f"--config {layout['linux_specedge_config']} --host {shlex.quote(specedge_bind_host)} --port {specedge_port}",
            "\n[node3：SpecEdge CPU adapter（official Tree/SpecExec/proactive semantics preserved）]",
            f"cd {repo} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(specedge_python)} {integration}/client_host.py "
                f"--config {layout['linux_specedge_config']}",
                environment=cpu_environment,
            ),
            "\n[node2：停止 FastSD target 后，以 vanilla 调度重启 target（同一端口/bind host）]",
            f"cd {repo} && CLOUD_SERVICE_HOST={shlex.quote(target_bind_host)} CLOUD_SERVICE_PORT={target_port} "
            f"FASTSD_TARGET_DEVICE={topology['standard_sd_target_device']} "
            f"{shlex.quote(target_python)} cloud/cloud_service.py "
            f"--target_model {models['target']} --draft_model {models['draft']} --dataset {dataset} "
            "--server_sched_mode vanilla",
            "\n[node3：标准投机解码（同样 CPU draft + 网络 + A6000 target）]",
            f"cd {repo} && "
            + prefixed_command(
                command_prefix,
                "bash scripts/run_vanilla_profile.sh "
                f"vanilla comparison/{run_id}/standard_sd {common} "
                f"--num_drafts {worker_count} --edge_gpus 1 "
                f"{'--edge_use_cpu' if cpu_draft else '--no-edge_use_cpu'} --edge_threads {draft_threads} "
                f"--arrival_distribution {config['dataset']['arrival_distribution']} "
                f"--arrival_rate {config['dataset']['arrival_rate_rps']} "
                f"--arrival_seed {config['dataset']['arrival_seed']}",
                environment={
                    **cpu_environment,
                    "PYTHON_BIN": python_bin,
                    "SERVER_URL": target_url,
                },
            ),
            "\n[node3：Draft-only（同一 CPU runtime；不访问 target）]",
            f"cd {repo} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(python_bin)} benchmark/eval_draft_pool.py --config {config_path}",
                environment=cpu_environment,
            ),
            "\n[node2：target-only greedy oracle（只用于 parity/quality reference，不纳入四方法）]",
            f"cd {repo} && {shlex.quote(target_python)} benchmark/eval_target_only.py --config {config_path}",
            "\n[网络拓扑说明：默认使用可覆盖的 target_host/port；受限网络可在 node3 建立双段 SSH 本地转发，"
            "让 127.0.0.1:18001 转发到 node2 127.0.0.1:18001。不要把转发命令或凭据写死到仓库。"
            "FastSD/SpecEdge 的请求耗时包含该网络路径，normalize 时保留 TTFT/E2E。]",
            "\n[node3：正式方法完成后记录 CPU 绑定/背景负载 postflight（保留 mpstat/load 快照）]",
            f"cd {repo} && "
            + prefixed_command(
                command_prefix,
                f"{shlex.quote(python_bin)} scripts/preflight_cpu.py "
                f"--threads {draft_threads} --expected-cpu-count {physical_cpu_count} "
                f"--expected-cpuset {shlex.quote(str(topology.get('fixed_cpuset', '')))} --phase post "
                f"--output {layout['linux_root']}/cpu_postflight.json "
                f"--commands-path {layout['linux_root']}/commands.txt",
                environment=cpu_environment,
            ),
            f"\n[method label: {'specedge_cpu_adapted' if cpu_draft else 'specedge'}; raw result expected at {specedge_raw}]",
        ]
    )
    print(plan)
    _append_command_record(layout["commands"], config_path, plan)
    append_status(
        layout["status"],
        method="suite",
        phase="plan",
        exit_code=0,
        command=f"{python_bin} scripts/eval_suite.py plan --config {config_path}",
    )
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
            "output_token_ids": item.get("output_token_ids", []),
            "workload_hash": item.get("workload_hash", manifest.get("workload_hash")),
            "network_rtt_ms": item.get("avg_transport_rtt_ms"),
            "network_bytes": item.get("network_bytes"),
            "kv_copy_ms": item.get("kv_copy_ms"),
            "kv_copy_bytes": item.get("kv_copy_bytes"),
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
        # SpecEdge warmup cycles use negative request IDs and are deliberately
        # excluded from the formal workload while remaining in raw evidence.
        if int(item.get("req_idx", 0)) < 0:
            continue
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
                "output_token_ids": completion.get("output_token_ids", []),
                "workload_hash": completion.get("workload_hash", manifest.get("workload_hash")),
                "reference": completion.get("reference"),
            }
        )
    return records, None


def normalize(config_path: str, method: str, input_path: str) -> int:
    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    source = Path(input_path)
    normalized_method = "specedge_cpu_adapted" if method == "specedge" else method
    if method == "fastsd":
        records, wallclock = normalize_fastsd(source, manifest)
    elif method in {"specedge", "specedge_cpu_adapted"}:
        records, wallclock = normalize_specedge(source, manifest)
    elif method == "standard_sd":
        records, wallclock = normalize_fastsd(source, manifest)
    elif method == "draft_only":
        records = _read_jsonl_files([source / "requests.jsonl"])
        wallclock = None
    else:
        raise ValueError(f"Unsupported method: {method}")
    output_dir = layout["local_root"] / "normalized" / normalized_method
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    write_text_once(
        requests_path,
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
    )
    if config["dataset"]["name"] == "humaneval":
        humaneval_path = output_dir / "humaneval_samples.jsonl"
        write_text_once(
            humaneval_path,
            "".join(
                json.dumps(
                    {
                        "task_id": record["sample_id"],
                        "completion": record.get("output_text", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
                for record in records
            ),
        )
    target_records = []
    target_path = layout["local_root"] / "target_only" / "requests.jsonl"
    if target_path.is_file():
        target_records = _read_jsonl_files([target_path])
    summary = summarize_requests(
        records,
        method=normalized_method,
        dataset=config["dataset"]["name"],
        workload_hash=manifest["workload_hash"],
        run_id=config["run_id"],
        wallclock_s=wallclock,
        evaluation_scope=manifest.get("evaluation_scope", "quality"),
        paired_records=target_records or None,
        extra={
            "draft_model": config["models"]["draft"],
            "target_model": config["models"]["target"],
            "source_method": method,
        },
    )
    write_json_once(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    append_status(
        layout["status"],
        method=normalized_method,
        phase="normalize",
        exit_code=0,
    )
    return 0


def compare(summary_paths: list[str], output: str | None) -> int:
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in summary_paths]
    hashes = {summary["workload_hash"] for summary in summaries}
    if len(hashes) != 1:
        raise ValueError(f"workload hashes differ: {sorted(hashes)}")
    columns = [
        "method", "evaluation_scope", "num_requests", "throughput_tok_s", "goodput_tok_s",
        "ttft_avg_ms", "ttft_p95_ms",
        "scheduled_ttft_p95_ms",
        "tpot_avg_ms", "tpot_p95_ms", "e2e_avg_ms", "accept_rate",
        "mean_accepted_tokens_per_verify", "quality_exact_match",
    ]
    rows = []
    for item in summaries:
        rows.append({
            "method": item["method"],
            "evaluation_scope": item.get("evaluation_scope", "unknown"),
            "num_requests": item["num_requests"],
            "throughput_tok_s": item["throughput_tok_s"],
            "goodput_tok_s": item.get("goodput_tok_s"),
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
        from io import StringIO

        buffer = StringIO()
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        write_text_once(output_path, buffer.getvalue())
    return 0


def parity(
    config_path: str,
    input_specs: list[str],
    *,
    reference: str = ORACLE_METHOD,
    output: str | None = None,
) -> int:
    """Create per-token parity JSONL from normalized/raw request files."""

    config = load_config(config_path)
    layout = output_layout(config)
    manifest = _read_manifest(config, layout)
    paths: dict[str, Path] = {}
    for spec in input_specs:
        if "=" not in spec:
            raise ValueError(f"parity input must be METHOD=PATH: {spec}")
        method, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        if path.is_dir():
            path = path / "requests.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[method] = path
    if reference not in paths:
        oracle_path = layout["local_root"] / ORACLE_METHOD / "requests.jsonl"
        if oracle_path.is_file():
            paths[reference] = oracle_path
    if reference not in paths:
        raise FileNotFoundError(
            f"reference {reference!r} is missing; pass --input {reference}=..."
        )
    records_by_method = {
        method: _read_jsonl_files([path]) for method, path in paths.items()
    }
    workload_hashes = {
        str(record.get("workload_hash"))
        for records in records_by_method.values()
        for record in records
        if record.get("workload_hash") is not None
    }
    if workload_hashes and workload_hashes != {manifest["workload_hash"]}:
        raise ValueError(
            f"parity workload hashes differ from manifest: {sorted(workload_hashes)}"
        )
    rows = build_token_parity_rows(records_by_method, reference_method=reference)
    parity_dir = layout["local_root"] / "parity"
    output_path = Path(output) if output else parity_dir / "token_parity.jsonl"
    summary_path = output_path.with_suffix(".summary.json")
    write_token_parity_report(
        rows,
        output_path,
        summary_path=summary_path,
        methods=paths.keys(),
        reference_method=reference,
    )
    append_status(
        layout["status"],
        method="suite",
        phase="parity",
        exit_code=0,
    )
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2))
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
    for command in ("prepare", "validate-models"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
    planner = subparsers.add_parser("plan")
    planner.add_argument("--config", required=True)
    planner.add_argument(
        "--python",
        dest="python_bin",
        help="explicit node3/specedge interpreter; recorded verbatim in commands.txt",
    )
    planner.add_argument(
        "--target-host",
        help="target host/IP or local-forward endpoint; topology is user-configurable",
    )
    planner.add_argument("--target-port", type=int, help="configurable target service port")
    planner.add_argument(
        "--cpu-prefix",
        help="optional raw taskset/cpuset/numactl prefix; recorded in commands.txt",
    )
    subparsers.add_parser(
        "validate-config",
        help="validate frozen latency/throughput topology and generation scope",
    ).add_argument("--config", required=True)
    normalizer = subparsers.add_parser("normalize")
    normalizer.add_argument("--config", required=True)
    normalizer.add_argument(
        "--method",
        required=True,
        choices=["fastsd", "specedge", "specedge_cpu_adapted", "standard_sd", "draft_only"],
    )
    normalizer.add_argument("--input", required=True)
    comparator = subparsers.add_parser("compare")
    comparator.add_argument("summaries", nargs="+")
    comparator.add_argument("--output")
    parity_parser = subparsers.add_parser("parity")
    parity_parser.add_argument("--config", required=True)
    parity_parser.add_argument("--input", action="append", required=True, dest="input_specs")
    parity_parser.add_argument("--reference", default=ORACLE_METHOD)
    parity_parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return prepare(args.config)
    if args.command == "plan":
        return print_plan(
            args.config,
            python_bin=args.python_bin,
            target_host=args.target_host,
            target_port=args.target_port,
            cpu_prefix=args.cpu_prefix,
        )
    if args.command == "normalize":
        return normalize(args.config, args.method, args.input)
    if args.command == "compare":
        return compare(args.summaries, args.output)
    if args.command == "parity":
        return parity(
            args.config,
            args.input_specs,
            reference=args.reference,
            output=args.output,
        )
    if args.command == "validate-models":
        return validate_models(args.config)
    if args.command == "validate-config":
        notes = validate_experiment_config(load_config(args.config))
        print(json.dumps({"valid": True, "notes": notes}, indent=2, ensure_ascii=False))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
