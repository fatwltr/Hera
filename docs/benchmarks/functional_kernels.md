# HERA functional kernel benchmarks

The independent benchmarks below measure the remaining projection,
conversion, and nonlinear functions. Run commands from the repository root.
The full build must have SEAL 4.4.5 and HERA-SCI prepared as described in
[README](../../README.md).

```bash
./scripts/bootstrap_dependencies.sh .deps 4
cmake --preset full
cmake --build build/full --target \
  hera_bench_he_functions hera_bench_mpc_functions hera_bench_conversion -j 2
ctest --test-dir build/full -R 'hera_bench_(he|mpc|conversion).*_smoke' \
  --output-on-failure
```

## HE functions

`hera_bench_he_functions` accepts `--kernel=qkv|wo|wu|wd|softmax-exp|gelu-poly`.
Each invocation loads the actual SEAL kernel. Context and keys, synthetic
input encryption, projection plaintext preparation, and final decryption are
outside the timer. Q/K/V share the same input but produce three outputs.
The benchmark uses the selected kernel's research-only level window, rather
than asserting that a standalone timing is an end-to-end Transformer latency.

```bash
./build/full/hera_bench_he_functions --kernel=qkv --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=wo  --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=wu  --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=wd  --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=softmax-exp \
  --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=gelu-poly \
  --warmup=1 --iterations=10
```

Options: `--tokens`, `--hidden`, `--heads`, `--kv-heads`, `--qk-width`,
`--value-width`, `--ffn-width`, `--degree`, `--he-threads`, `--warmup`,
`--iterations`. Defaults are deliberately tiny. `--degree` is CKKS polynomial
degree; larger shapes can require significantly more RAM and keys. The HE
Softmax benchmark measures only the exponent polynomial, not MPC row maximum
or normalization. The GeLU HE benchmark measures both polynomial branches,
not MPC compare/mux. QKV/Wo/Wu/Wd use prepared model plaintexts, so the
reported duration represents online evaluation rather than first-time weight
encoding. The output includes HE operation counts and an untimed final
numerical check.

## Two-party MPC functions

`hera_bench_mpc_functions` forks a local client and server. Each command
measures real SCI/VOLE protocol calls on pre-created additive shares. The
first iteration can initialize an MPC session; use `--warmup=1` or more to
exclude that from the reported online mean. Both parties' time, communication,
and rounds are reported, along with a final untimed reconstruction check.

```bash
./build/full/hera_bench_mpc_functions --kernel=softmax-max \
  --tokens=2 --heads=1 --warmup=1 --iterations=10 --port=24900
./build/full/hera_bench_mpc_functions --kernel=softmax-normalize \
  --tokens=2 --heads=1 --warmup=1 --iterations=10 --port=24910
./build/full/hera_bench_mpc_functions --kernel=gelu-select \
  --tokens=2 --width=8 --warmup=1 --iterations=10 --port=24920
```

`softmax-max` is `CenterAndMask` (row max using compare/mux, then shifted
scores); `softmax-normalize` is `ClipAndNormalize` (threshold/mux, row sum,
reciprocal, secure multiply/truncate). `gelu-select` is the MPC compare/mux
stage after the HE F0/F1 polynomials. Inputs are generated before timing and
are not encrypted or converted inside these boundaries. The benchmarks use
the no-mask path; padding-mask correctness remains covered by component tests.

## CKKS↔MPC conversion

`hera_bench_conversion` uses the existing two-party protocol and reports
CKKS→MPC and MPC→CKKS timings separately. Public key exchange, input
encryption, and final ciphertext transfer/decryption are outside both timers;
A/Z masking, communication, SCI field/ring operations, and fixed-point
transforms remain inside. Select a profile from the JSON configuration.

```bash
./build/full/hera_bench_conversion \
  --party=local --config=configs/conversion_smoke.json \
  --profile=conversion_test --port=24930 --threads=1 \
  --warmup=1 --iterations=10
```

For two machines, run the same configuration and counts with
`--party=server` on one and `--party=client --ip=<server-ip>` on the other.
The CLI currently uses a fixed eight-logical-value, one-ciphertext complex
sample per iteration. Its throughput must not be extrapolated to large
batches without a separate batch-size sweep. The final round-trip is
checked against the repository's current 48 scale-13-ULP smoke bound; this
does not claim the stricter four-ULP target.

All these HE profiles set `security_level=none` and are for research
measurement only. Do not compare their times with BLB paper latencies unless
shape, level chain, hardware, thread count, and timer boundaries match.
