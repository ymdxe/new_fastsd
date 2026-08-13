"""Evaluate a pool of draft-only workers on the canonical comparison workload."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:  # pragma: no cover - optional for non-GPTQ runs
    AutoGPTQForCausalLM = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.common_metrics import summarize_requests, write_json
from src.evaluation import load_canonical_jsonl
from src.util import norm_logits, sample, seed_everything


def _encode_prompt(tokenizer, prompt: str, dataset: str) -> torch.Tensor:
    if dataset != "mt_bench":
        return tokenizer.encode(prompt, return_tensors="pt")

    messages = [{"role": "user", "content": prompt}]
    template_args = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            enable_thinking=False,
            **template_args,
        )
    except TypeError:
        input_ids = tokenizer.apply_chat_template(messages, **template_args)
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    return input_ids


def _load_model(model_path: str, device: str):
    if (Path(model_path) / "quantize_config.json").is_file():
        if AutoGPTQForCausalLM is None:
            raise RuntimeError("auto_gptq is required for a quantized draft model")
        return AutoGPTQForCausalLM.from_quantized(
            model_path,
            device=device,
            use_safetensors=True,
            trust_remote_code=True,
            use_triton=False,
        ).eval()
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval()


@torch.inference_mode()
def _generate(model, input_ids, tokenizer, generation: dict[str, Any]) -> tuple[torch.Tensor, dict]:
    device = model.device
    output_ids = input_ids.to(device)
    prompt_len = output_ids.shape[1]
    past_key_values = None
    start = time.perf_counter()
    first_token_time = None

    for _ in range(int(generation["max_new_tokens"])):
        model_input = output_ids if past_key_values is None else output_ids[:, -1:]
        outputs = model(model_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        probabilities = norm_logits(
            outputs.logits[:, -1, :],
            float(generation["temperature"]),
            int(generation["top_k"]),
            float(generation["top_p"]),
        )
        next_token = sample(probabilities)
        output_ids = torch.cat((output_ids, next_token), dim=1)
        if first_token_time is None:
            torch.cuda.synchronize(device)
            first_token_time = time.perf_counter()
        if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize(device)
    end = time.perf_counter()
    generated_tokens = int(output_ids.shape[1] - prompt_len)
    if first_token_time is None:
        first_token_time = end
    metrics = {
        "generated_tokens": generated_tokens,
        "ttft_ms": (first_token_time - start) * 1000.0,
        "tpot_ms": (
            (end - first_token_time) * 1000.0 / (generated_tokens - 1)
            if generated_tokens > 1
            else 0.0
        ),
        "e2e_ms": (end - start) * 1000.0,
    }
    return output_ids, metrics


def _worker(
    worker_idx: int,
    device: str,
    records: list[dict[str, Any]],
    model_path: str,
    generation: dict[str, Any],
    output_path: str,
    barrier,
    start_epoch,
    workload_hash: str,
    num_workers: int,
) -> None:
    model = _load_model(model_path, device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    barrier.wait()
    if worker_idx == 0:
        start_epoch.value = time.time()
    barrier.wait()

    with Path(output_path).open("w", encoding="utf-8") as output:
        for record in records[worker_idx::num_workers]:
            scheduled = float(record.get("scheduled_arrival_s", 0.0))
            remaining = float(start_epoch.value) + scheduled - time.time()
            if remaining > 0:
                time.sleep(remaining)
            actual_arrival = max(0.0, time.time() - float(start_epoch.value))
            request_start = time.perf_counter()
            input_ids = _encode_prompt(tokenizer, record["prompt"], record["dataset"])
            seed_everything(int(generation["seed"]) + int(record["global_index"]))
            generated, model_metrics = _generate(
                model, input_ids, tokenizer, generation
            )
            request_end = time.perf_counter()
            generated_ids = generated[0, input_ids.shape[1] :]
            request_e2e_ms = (request_end - request_start) * 1000.0
            ttft_ms = model_metrics["ttft_ms"] + (
                request_e2e_ms - model_metrics["e2e_ms"]
            )
            payload = {
                "schema_version": 1,
                "sample_id": record["sample_id"],
                "global_index": int(record["global_index"]),
                "worker_idx": worker_idx,
                "dataset": record["dataset"],
                "method": "draft_only",
                "workload_hash": workload_hash,
                "scheduled_arrival_s": scheduled,
                "actual_arrival_s": actual_arrival,
                "completion_s": max(0.0, time.time() - float(start_epoch.value)),
                "arrival_lag_ms": max(0.0, (actual_arrival - scheduled) * 1000.0),
                "generated_tokens": int(model_metrics["generated_tokens"]),
                "ttft_ms": ttft_ms,
                "tpot_ms": float(model_metrics["tpot_ms"]),
                "e2e_ms": request_e2e_ms,
                "output_text": tokenizer.decode(generated_ids, skip_special_tokens=True),
                "reference": record.get("reference"),
            }
            output.write(json.dumps(payload, ensure_ascii=False) + "\n")
            output.flush()


def run(config_path: str) -> int:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    run_root = REPO_ROOT / "exp" / "comparison" / config["run_id"]
    canonical_path = run_root / "inputs" / "canonical.jsonl"
    manifest_path = run_root / "run_manifest.json"
    if not canonical_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("run scripts/eval_suite.py prepare before draft-only evaluation")
    records = load_canonical_jsonl(canonical_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    devices = list(config["topology"]["specedge_draft_devices"])
    if not devices:
        raise ValueError("topology.specedge_draft_devices must contain at least one device")
    output_dir = run_root / "draft_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(devices))
    start_epoch = ctx.Value("d", 0.0)
    processes = []
    output_paths = []
    for worker_idx, device in enumerate(devices):
        output_path = output_dir / f"requests_worker{worker_idx}.jsonl"
        output_paths.append(output_path)
        process = ctx.Process(
            target=_worker,
            args=(
                worker_idx,
                device,
                records,
                config["models"]["draft"],
                config["generation"],
                str(output_path),
                barrier,
                start_epoch,
                manifest["workload_hash"],
                len(devices),
            ),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    failed = [process.exitcode for process in processes if process.exitcode != 0]
    if failed:
        raise RuntimeError(f"{len(failed)} draft worker(s) failed: {failed}")

    request_records = []
    for path in output_paths:
        with path.open("r", encoding="utf-8") as handle:
            request_records.extend(json.loads(line) for line in handle if line.strip())
    request_records.sort(key=lambda item: int(item["global_index"]))
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as output:
        for record in request_records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = summarize_requests(
        request_records,
        method="draft_only",
        dataset=config["dataset"]["name"],
        workload_hash=manifest["workload_hash"],
        run_id=config["run_id"],
        extra={"draft_model": config["models"]["draft"], "num_workers": len(devices)},
    )
    write_json(summary, output_dir / "common_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.config))
