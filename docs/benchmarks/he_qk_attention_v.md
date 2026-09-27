# QK and Attention×V kernel benchmarks

These are two **HE-only** benchmarks. The timer starts with the required
Q/K or Softmax-A/V ciphertexts already encrypted in each kernel's native
packing, and stops when the HE output is produced. Context/key generation,
packing, encoding/encryption of inputs, and final decryption/accuracy checks
are outside the timer. There is no CKKS↔MPC conversion, projection, Softmax,
Wo, or network traffic in the measured interval. Each iteration reuses the
same input ciphertexts and Galois keys; output correctness is checked after
the last timed iteration.

Build both benchmarks with the full preset:

```bash
cmake --preset full
cmake --build build/full --target hera_bench_qk hera_bench_attention_v -j 2
ctest --test-dir build/full -R 'hera_bench_(qk|attention_v)_smoke' --output-on-failure
```

QK alone can also be built with `HERA_ENABLE_HE=ON`; Attention×V currently
links the Attention/LN2 stage library and its crypto dependencies.

Run both at the same logical shape and concurrency:

```bash
./build/full/hera_bench_qk \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2

./build/full/hera_bench_attention_v \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2
```

`--width` is Q/K head width for QK and V head width for Attention×V. Q is
pre-scaled by `1/sqrt(width)` before encryption, matching the production QK
contract. Attention A is real-only and V uses real/imag value-column pairs.
The synthetic inputs are deterministic but encryption uses fresh randomness.

These programs deliberately use independent, **research-only** CKKS chains:

| Kernel | CKKS data-prime bits | Input → output level | Measured core |
| --- | --- | --- | --- |
| QK | `[60,60,60,60,60,60]` | L0 → L5 | QK preprocessing, ciphertext products, diagonal recovery/pairing |
| Attention×V | `[60,60,60,60]` | L0 → L3 | A preprocessing, V preprocessing, ciphertext products/recovery |

The registry appends a separate key-switch special prime. Neither chain is
intended to match a whole HERA Transformer block or assert 128-bit security;
both set `security_level=none`. The benchmark fails if the output does not
reach its terminal data-prime level or exceeds the numerical error check.

## BLB source comparison and timing caveat

The local BLB mirror contains the relevant functions in
`SCI/tests/bert_large_bolt/linear_ckks.cpp`:

| Operation | BLB source | HERA source | Shared idea / important difference |
| --- | --- | --- | --- |
| QK | `bert_cipher_cipher_cross_packing` | `MultiHeadQkKernel::Evaluate` | Both use encrypted Q/K and diagonal-style products. BLB is specialized to compile-time BERT/GPT shape macros and pairs heads in complex slots; HERA has runtime head tiling, GQA/MQA mapping and a different native packing/recovery schedule. |
| Attention×V | `softmax_v` | `MultiHeadAttentionValueKernel::Evaluate` | Both start from encrypted A and V and perform preprocessing followed by ciphertext products and accumulation. HERA uses dynamic baby/giant groups and real/imag V columns; the BLB function uses fixed `PACKING_NUM` and `INPUT_DIM` macros. |

The BLB source tree does **not** include a built standalone kernel executable
on this checkout. Its existing `qk time` measurement begins before a mask
plaintext is encoded and accepts the Q/K/V ciphertext vector **by value**;
its `softmax_v total time` starts after a mask encode but before its core
loops. Both have fixed-shape packing and extra simulation steps. Consequently
those printed durations, or the paper's end-to-end/table latencies, are not
an apples-to-apples native BLB baseline for the boundary above.

If an independently built BLB benchmark is instrumented to start with its
own correctly packed input ciphertexts, stop at its HE output, and run on the
same hardware/shape/thread budget, pass its mean time via `--blb-ms=...` to
the corresponding HERA command. The program then prints the external value
and `HERA_OVER_BLB_TIME_RATIO`. It intentionally prints
`BLB_EXTERNAL_MEAN_MS=NOT_MEASURED` otherwise; it never substitutes a paper
latency or a guessed number. The ratio is only meaningful if the reported
boundary and key/context setup exclusions match.

## Example HERA-only local result

On the developer's 16-logical-CPU / 31-GiB local machine, Release SEAL
4.4.5, `tokens=32, heads=8, width=16, degree=8192, warmup=1,
iterations=3, he-threads=2`:

| Kernel | Mean HE time | Rotations | Ciphertext×ciphertext products | Output level |
| --- | ---: | ---: | ---: | ---: |
| QK | 980.4 ms | 572 | 32 | L5 |
| Attention×V | 757.1 ms | 63 | 32 | L3 |

These numbers are not a BLB speedup claim. The checked-out BLB source defaults
to `INPUT_DIM=128`, `PACKING_NUM=12`, `COMMON_DIM=768` (64 columns per head).
For that shape, use `--tokens=128 --heads=12 --width=64 --degree=32768`
and collect a matched BLB-native result separately; this is substantially
more expensive. Its commented 16-head/1024-width variant requires changing
BLB's compile-time macros before a BERT-large comparison.
