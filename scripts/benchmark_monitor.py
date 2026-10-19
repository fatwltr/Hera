#!/usr/bin/env python3
"""Resource monitoring shared by the optional native BLB benchmarks."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def monitored(commands, env, directory, as_gib=16, timeout=1800):
    directory.mkdir()

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (as_gib * 1024**3, as_gib * 1024**3))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    files, processes = [], []
    started = time.monotonic()
    peak = swap_peak = 0
    minimum = 1 << 60
    reason = None
    try:
        for i, cmd in enumerate(commands):
            log = (directory / (f"process{i+1}.log")).open("w")
            files.append(log)
            processes.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=limits, start_new_session=True))
            if len(commands) > 1 and i == 0:
                time.sleep(.2)
        with (directory / "telemetry.jsonl").open("w") as telemetry:
            while True:
                pids = []
                groups = {p.pid for p in processes}
                for path in Path("/proc").iterdir():
                    if not path.name.isdigit():
                        continue
                    try:
                        if os.getpgid(int(path.name)) in groups:
                            pids.append(int(path.name))
                    except ProcessLookupError:
                        pass
                rss = swap = 0
                for pid in pids:
                    try:
                        fields = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
                        rss += int(fields.get("VmRSS", "0 kB").split()[0]) * 1024
                        swap += int(fields.get("VmSwap", "0 kB").split()[0]) * 1024
                    except FileNotFoundError:
                        pass
                mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available = int(mem["MemAvailable"].split()[0]) * 1024
                peak, swap_peak, minimum = max(peak, rss), max(swap_peak, swap), min(minimum, available)
                elapsed = time.monotonic() - started
                codes = [p.poll() for p in processes]
                telemetry.write(json.dumps({"seconds": elapsed, "rss": rss, "swap": swap, "available": available, "exits": codes}) + "\n")
                telemetry.flush()
                if all(code is not None for code in codes):
                    break
                if any(code is not None and code != 0 for code in codes):
                    reason = "peer_failed"
                if available < 1024**3 or swap:
                    reason = "resource_guard"
                if elapsed > timeout:
                    reason = "timeout"
                if reason:
                    break
                time.sleep(.5)
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        for log in files:
            log.close()
    result = {"commands": commands, "environment": {k: env[k] for k in ["OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_MAX_ACTIVE_LEVELS", "HERA_RUNTIME_TRACE"] if k in env}, "exit_codes": [p.returncode for p in processes], "reason": reason, "wall_s": time.monotonic() - started, "peak_rss_bytes": peak, "peak_swap_bytes": swap_peak, "minimum_available_bytes": minimum, "as_gib_per_process": as_gib}
    (directory / "run.json").write_text(json.dumps(result, indent=2))
    if reason or any(p.returncode != 0 for p in processes):
        raise RuntimeError(f"benchmark failed in {directory}: {reason}, exits={result['exit_codes']}")

