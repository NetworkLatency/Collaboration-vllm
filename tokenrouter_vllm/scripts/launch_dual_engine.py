#!/usr/bin/env python3
"""Launch cooperative SLM + LLM vLLM v0.33.0 engines from a YAML config."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SLM_SCHEDULER_CLS = "vllm.v1.core.sched.token_routing.GlimpRouterSLMScheduler"
LLM_SCHEDULER_CLS = "vllm.v1.core.sched.token_routing.GlimpRouterLLMScheduler"


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "PyYAML is required to load YAML config files: pip install pyyaml"
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping at top level of {path}")
    return data


def build_engine_command(
    role: str,
    cfg: dict[str, Any],
    *,
    vllm_bin: str,
) -> tuple[list[str], dict[str, str]]:
    routing_cfg = dict(cfg.get("routing", {}))
    transport_cfg = dict(cfg.get("transport", {}))
    slm_cfg = cfg["slm"]
    llm_cfg = cfg["llm"]

    if role == "slm":
        engine_cfg = slm_cfg
        scheduler_cls = SLM_SCHEDULER_CLS
        model_name = str(slm_cfg.get("model_name", "slm"))
        peer_name = str(llm_cfg.get("model_name", "llm"))
        is_entry_model = True
        peer_bind = transport_cfg["slm_bind_address"]
        peer_connect = transport_cfg["llm_bind_address"]
    elif role == "llm":
        engine_cfg = llm_cfg
        scheduler_cls = LLM_SCHEDULER_CLS
        model_name = str(llm_cfg.get("model_name", "llm"))
        peer_name = str(slm_cfg.get("model_name", "slm"))
        is_entry_model = False
        peer_bind = transport_cfg["llm_bind_address"]
        peer_connect = transport_cfg["slm_bind_address"]
    else:
        raise ValueError(f"Unknown role: {role!r}")

    token_router_cfg: dict[str, Any] = {
        **routing_cfg,
        "model_name": model_name,
        "peer_name": peer_name,
        "is_entry_model": is_entry_model,
        "peer_bind_address": peer_bind,
        "peer_connect_address": peer_connect,
    }
    additional_config = json.dumps(
        {"token_router": token_router_cfg}, separators=(",", ":")
    )

    cmd = [
        vllm_bin,
        "serve",
        str(engine_cfg["model"]),
        "--host",
        str(engine_cfg.get("host", "127.0.0.1")),
        "--port",
        str(engine_cfg["port"]),
        "--tensor-parallel-size",
        str(engine_cfg.get("tensor_parallel_size", 1)),
        "--gpu-memory-utilization",
        str(engine_cfg.get("gpu_memory_utilization", 0.85)),
        "--max-model-len",
        str(engine_cfg.get("max_model_len", 16384)),
        "--max-num-seqs",
        str(engine_cfg.get("max_num_seqs", 64)),
        "--dtype",
        str(engine_cfg.get("dtype", "auto")),
        "--scheduler-cls",
        scheduler_cls,
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--additional-config",
        additional_config,
    ]
    served_name = engine_cfg.get("served_model_name")
    if served_name:
        cmd.extend(["--served-model-name", str(served_name)])
    for extra in engine_cfg.get("extra_args", []) or []:
        cmd.append(str(extra))

    env = os.environ.copy()
    env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    if "cuda_visible_devices" in engine_cfg:
        env["CUDA_VISIBLE_DEVICES"] = str(engine_cfg["cuda_visible_devices"])
    return cmd, env


def wait_for_health(
    name: str,
    port: int,
    proc: subprocess.Popen[bytes],
    *,
    timeout_s: float,
) -> None:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            raise RuntimeError(f"{name} exited prematurely with code {rc}")
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status == 200:
                    print(f"[launcher] {name} healthy at {url}", flush=True)
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for {name} health at {url}")


def terminate_processes(procs: list[tuple[str, subprocess.Popen[bytes]]]) -> None:
    for name, proc in procs:
        if proc.poll() is None:
            print(f"[launcher] Stopping {name} (pid={proc.pid})...", flush=True)
            try:
                if hasattr(os, "killpg"):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 15.0
    for name, proc in procs:
        remaining = max(0.1, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            print(f"[launcher] Force-killing {name} (pid={proc.pid})", flush=True)
            if hasattr(os, "killpg"):
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Launch TokenRouter x vLLM dual-engine server."
    )
    default_cfg = (
        Path(__file__).resolve().parent.parent / "configs" / "glimprouter_qwen3.yaml"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_cfg,
        help="Path to YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--vllm-bin",
        default="vllm",
        help="Path to vllm CLI binary (default: vllm).",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for each engine /health endpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print launch commands and env overrides without starting processes.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm_cmd, llm_env = build_engine_command("llm", cfg, vllm_bin=args.vllm_bin)
    slm_cmd, slm_env = build_engine_command("slm", cfg, vllm_bin=args.vllm_bin)

    if args.dry_run:
        for role, cmd, env in (("llm", llm_cmd, llm_env), ("slm", slm_cmd, slm_env)):
            gpus = env.get("CUDA_VISIBLE_DEVICES", "<unset>")
            print(
                f"# [{role.upper()}] CUDA_VISIBLE_DEVICES={gpus} "
                f"VLLM_USE_V2_MODEL_RUNNER=0"
            )
            print(" ".join(shlex.quote(part) for part in cmd))
            print()
        return 0

    popen_kwargs: dict[str, Any] = {}
    if hasattr(os, "setsid"):
        popen_kwargs["start_new_session"] = True

    procs: list[tuple[str, subprocess.Popen[bytes]]] = []
    shutting_down = False

    def _handle_signal(signum: int, _frame: Any) -> None:
        nonlocal shutting_down
        if not shutting_down:
            shutting_down = True
            print(
                f"\n[launcher] Received signal {signum}, shutting down...", flush=True
            )
            terminate_processes(procs)
            sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        print("[launcher] Starting LLM peer engine...", flush=True)
        llm_proc = subprocess.Popen(llm_cmd, env=llm_env, **popen_kwargs)
        procs.append(("llm", llm_proc))

        print("[launcher] Starting SLM entry engine...", flush=True)
        slm_proc = subprocess.Popen(slm_cmd, env=slm_env, **popen_kwargs)
        procs.append(("slm", slm_proc))

        wait_for_health(
            "llm", int(cfg["llm"]["port"]), llm_proc, timeout_s=args.startup_timeout
        )
        wait_for_health(
            "slm", int(cfg["slm"]["port"]), slm_proc, timeout_s=args.startup_timeout
        )

        slm_port = int(cfg["slm"]["port"])
        served_name = cfg["slm"].get("served_model_name", cfg["slm"]["model"])
        print(
            f"[launcher] Dual-engine service ready! "
            f"OpenAI endpoint: http://127.0.0.1:{slm_port}/v1 (model={served_name})",
            flush=True,
        )

        while True:
            for name, proc in procs:
                rc = proc.poll()
                if rc is not None:
                    print(
                        f"[launcher] {name} exited with code {rc}; stopping peer.",
                        flush=True,
                    )
                    terminate_processes(procs)
                    return rc
            time.sleep(1.0)
    except Exception:
        terminate_processes(procs)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
