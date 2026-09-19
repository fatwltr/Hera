# runtime_tiny local benchmark (2026-09-19)

This is a functional protocol-overhead benchmark, not a BERT-size benchmark.

## Environment

- CPU: Intel Core i5-14400F, 10 cores / 16 logical CPUs
- RAM: 31 GiB
- OS: Ubuntu 22.04, Linux 6.8.0-138-generic
- Compiler: GCC 9.5.0
- SEAL: 4.4.5
- HERA commit: `0248117`
- Parties: two local processes over loopback
- Threads: 1
- Warmup/repeat: 0/1
- Final verification/decryption: disabled
- Shape: tokens=2, model_width=4, q_heads=2, kv_heads=1,
  qk_head_width=2, value_head_width=2, ffn_width=5

Command:

```bash
/usr/bin/time -v ./build/repro-full/hera_private_inference \
  --config=configs/runtime_tiny.json \
  --party=local --port=25100 --threads=1 \
  --input-seed=1 --weight-seed=2 --warmup=0 --repeat=1
```

## End-to-end results

| Metric | Client | Server |
|---|---:|---:|
| Setup | 5607.47 ms | 5750.34 ms |
| Model plaintext Prepare | 0.001 ms | 142.855 ms |
| Prepared plaintexts | 0 | 73 |
| Prepared bytes | 0 | 7,919,896 |
| Online | 2536.00 ms | 2393.15 ms |
| Output transfer | 0.078 ms | 0.065 ms |
| Reported communication | 113,707,547 B | 13,598,973 B |
| Rounds | 903 | 903 |

GNU time reported 8.22 s wall time, 6.42 s user CPU, 3.32 s system CPU,
118% average CPU utilization, and 4,020,540 KiB maximum resident set size.

## Online stage breakdown

The latency column uses the slower party for each synchronized stage. Combined
communication is the sum of the two party-local counters; it is not a packet
capture.

| Stage | Latency | Share | Combined communication | Rounds | Conversions |
|---|---:|---:|---:|---:|---:|
| Pre-LN entry + Block 2 | 771.225 ms | 31.47% | 39.97 MiB | 197 | 6 |
| Softmax | 612.775 ms | 25.00% | 20.88 MiB | 279 | 3 |
| Attention×V + Wo + LN2 | 474.323 ms | 19.36% | 20.67 MiB | 218 | 3 |
| LN2 tail + Wu + GeLU | 524.479 ms | 21.40% | 33.54 MiB | 168 | 3 |
| Wd + residual + realify | 67.838 ms | 2.77% | 6.35 MiB | 41 | 1 |
| Total staged work | 2450.640 ms | 100% | 121.41 MiB | 903 | 16 |

Server HE operation totals were 116 rotations, 91 ciphertext-plaintext
multiplications, 16 ciphertext-ciphertext multiplications, 15
relinearizations and 43 rescale operations. MPC operation count was 216.

## Interpretation

- Setup/key generation dominates a single inference: about 5.75 s versus a
  2.54 s observed online critical path. Reusing a session amortizes this cost.
- Pre-LN/Block 2 is the largest online latency and communication stage.
- Softmax has the most protocol rounds (279), despite no reported HE rotations.
- FFN/GeLU is the second-largest communication consumer.
- Block 5 is comparatively small in both latency and communication.
- These proportions include fixed protocol overheads at tiny dimensions and
  must not be extrapolated linearly to the 12-layer, N=32768 configuration.
