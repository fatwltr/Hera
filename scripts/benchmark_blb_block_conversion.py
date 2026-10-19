#!/usr/bin/env python3
"""Measure the native BLB conversion workload for its BERT-base block shapes.

This replays valid synthetic conversion boundaries; it does not run the
artifact's incomplete BERT semantics. Outer ciphertext operations are excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

from benchmark_monitor import monitored

ROOT = Path(__file__).resolve().parents[1]


def summarize(out):
    result = {"scope": "two-party conversion workload replay; FFT/NTT/RNS and ring protocols; "
              "excludes outer encryption/decryption, ciphertext transport, model HE and nonlinear MPC",
              "model": {"tokens": 128, "width": 768, "heads": 12, "ffn": 3072},
              "mpc_ring": 40, "mpc_scale": 20, "kappa": 40, "workers": 6, "runs": {}}
    for name in ["bounded", "general", "bounded_dual"]:
        directory = out / name
        if not (directory / 'run.json').exists():
            continue
        calls, samples = [], []
        for line in (directory / 'process1.log').read_text().splitlines():
            if not line.startswith(('BLB_BLOCK_CALL,', 'BLB_BLOCK_SAMPLE,')):
                continue
            fields = dict(part.split('=', 1) for part in line.split(',')[1:])
            for key, value in list(fields.items()):
                try:
                    fields[key] = float(value) if any(c in value.lower() for c in '.e') else int(value)
                except ValueError:
                    pass
            (calls if line.startswith('BLB_BLOCK_CALL,') else samples).append(fields)
        measured = [s for s in samples if not s['warmup']]
        if len(calls) != 17 or not measured or any(c['status'] != 'PASS' for c in calls):
            raise RuntimeError(f'incomplete or incorrect benchmark: {directory}')
        base = [c for c in calls if not c['supplemental']]
        extra = [c for c in calls if c['supplemental']]
        entry = {'calls': calls, 'samples': samples,
                 'base_calls': len(base), 'supplement_calls': len(extra),
                 'base_ms': statistics.mean(s['base_ms'] for s in measured),
                 'base_median_ms': statistics.median(s['base_ms'] for s in measured),
                 'base_stdev_ms': statistics.stdev(s['base_ms'] for s in measured) if len(measured) > 1 else 0,
                 'base_bytes': statistics.mean(s['base_bytes'] for s in measured),
                 'supplement_ms': statistics.mean(s['supplement_ms'] for s in measured),
                 'supplement_bytes': statistics.mean(s['supplement_bytes'] for s in measured),
                 'max_error_ulp': max(c['max_error_ulp'] for c in calls),
                 'resources': json.loads((directory / 'run.json').read_text())}
        for direction in ['C2M', 'M2C']:
            group = [c for c in base if c['direction'] == direction]
            entry[direction] = {'calls': len(group), 'ms': sum(c['mean_ms'] for c in group),
                                'bytes': sum(c['both_bytes'] for c in group),
                                'ciphertexts': sum(c['ciphertexts'] for c in group),
                                'coefficients': sum(c['ciphertexts'] * c['degree'] for c in group),
                                'active_values': sum(c['active'] for c in group),
                                'capacity_values': sum(c['capacity'] for c in group)}
        entry['separate_branch_assumption_ms'] = entry['base_ms'] + entry['supplement_ms']
        entry['separate_branch_assumption_bytes'] = entry['base_bytes'] + entry['supplement_bytes']
        result['runs'][name] = entry
        print(f"{name}: base14={entry['base_ms']:.3f} ms; supplements3={entry['supplement_ms']:.3f} ms; "
              f"sum17={entry['separate_branch_assumption_ms']:.3f} ms; "
              f"base_comm={entry['base_bytes']/1e6:.3f} MB; error<={entry['max_error_ulp']:.3f} ULP", flush=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native', type=Path, default=ROOT.parent / 'blb/SCI')
    parser.add_argument('--reuse-build', action='store_true')
    parser.add_argument('--reuse-native-build', action='store_true')
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--modes', nargs='+', choices=['bounded', 'general', 'bounded_dual'],
                        default=['bounded', 'general', 'bounded_dual'])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    out, native = args.output.resolve(), args.native.resolve()
    if args.summarize_only:
        summarize(out)
        return
    out.mkdir(parents=True, exist_ok=True)
    binary = out / 'blb_block_conversion_benchmark'
    build = out / 'native_build'
    if not args.reuse_build:
        # Rebuild current native bridge sources and the isolated adapter.
        if not args.reuse_native_build:
            with (out / 'native_build.log').open('w') as log:
                subprocess.run([sys.executable, str(ROOT/'scripts/benchmark_blb_mm.py'),
                                '--native', str(native), '--output', str(build), '--build-only'],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
        commands = json.loads((build / 'build_commands.json').read_text())
        compile_command = next(c for c in commands if str(build / 'bridge_driver.o') in c and '-c' in c)
        compile_command = [str(ROOT / 'tests/blb_block_conversion_benchmark.cpp')
                           if p == str(ROOT/'tests/blb_mm_bridge_benchmark.cpp')
                           else str(out / 'block_driver.o') if p == str(build/'bridge_driver.o') else p
                           for p in compile_command]
        link_command = next(c for c in commands if str(build / 'blb_mm_bridge_benchmark') in c and '-c' not in c)
        link_command = [str(out / 'block_driver.o') if p == str(build/'bridge_driver.o')
                        else str(binary) if p == str(build/'blb_mm_bridge_benchmark') else p for p in link_command]
        for name, command in [('block_driver', compile_command), ('block_link', link_command)]:
            with (out / (name + '.build.log')).open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        (out / 'block_build_commands.json').write_text(json.dumps([compile_command, link_command], indent=2)+'\n')
        snapshot = out / 'source_snapshot'
        snapshot.mkdir(exist_ok=True)
        source = native / 'tests/bert_large_bolt'
        paths = [ROOT/'tests/blb_block_conversion_benchmark.cpp', Path(__file__),
                 ROOT/'scripts/benchmark_blb_mm.py', ROOT/'scripts/benchmark_monitor.py',
                 *[source/n for n in ['ckks_bert.cpp', 'ckks_bert_basic.cpp', 'ckks_bert_large_main.cpp',
                                     'linear_ckks.cpp', 'linear_ckks.h', 'nonlinear.cpp', 'nonlinear.h',
                                     'he.cpp', 'he.h']], *source.glob('mm_*')]
        for p in paths:
            shutil.copy2(p, snapshot / p.name)
        (out / 'source_manifest.json').write_text(json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, indent=2)+'\n')
        (out / 'binary.sha256').write_text(hashlib.sha256(binary.read_bytes()).hexdigest()+'\n')
    if args.build_only:
        return
    for name in args.modes:
        env = dict(os.environ, OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2', OMP_MAX_ACTIVE_LEVELS='1',
                   BLB_PUBLIC_QUARTER_RANGE='0' if name == 'general' else '1',
                   BLB_CORRECT_CONVERSION='1', BLB_CONVERSION_THREADS='6')
        commands = [['taskset', '-c', '0,2,4,6,8,10', str(binary), str(party), '28500',
                     str(int(name == 'bounded_dual')), str(args.warmup), str(args.repeats)] for party in [1,2]]
        monitored(commands, env, out/name, as_gib=6, timeout=600)
        (out/name/'conversion_environment.json').write_text(json.dumps(
            {k: v for k,v in env.items() if k.startswith(('BLB_', 'OMP_'))}, indent=2)+'\n')
        print(name + ' completed with all boundary oracles PASS', flush=True)
    summarize(out)


if __name__ == '__main__':
    main()
