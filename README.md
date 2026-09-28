# HERA

HERA 是 CKKS/MPC 私有 Transformer 推理原型，支持多层 Pre-LN + GeLU、
MHA/GQA/MQA 和可变宽度。输入、输出均为 CKKS 密文。当前配置允许
`security_level=none`，**仅供研究，不能用于生产安全部署**。

## 在 Ubuntu 服务器上构建并验证

Ubuntu 22.04/24.04、x86-64；需访问 GitHub/GitLab，并具备仓库克隆权限：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync \
  libssl-dev libgmp-dev libmpfr-dev libboost-dev

git clone git@github.com:fatwltr/Hera.git
cd Hera

./scripts/bootstrap_dependencies.sh .deps 2
./scripts/build_and_test.sh full 2 .deps

./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json \
  --party=local --verify --port=23100
```

依赖安装在 `.deps/`，无需预设 SEAL/SCI 路径。`2` 为并行编译任务数。
成功标志：`HERA_BUILD_STATUS=PASS`、`HERA_PRIVATE_INFERENCE_STATUS=PASS`。
本机双进程运行需保证端口 23100 起的协议端口空闲。

仅下载 SEAL、BLB、BOLT 和 Eigen 源码（不编译）：

```bash
./scripts/download_dependencies.sh .deps
```

仅检查图框架（无需 SEAL/SCI）：

```bash
./scripts/build_and_test.sh framework 2
./build/framework/hera_private_inference \
  --config=configs/figure10_smoke.json --dry-run --dump-graph
```

## 单算子 benchmark

只构建和测试 benchmark：

```bash
cmake --preset full
cmake --build build/full --target \
  hera_bench_qk hera_bench_attention_v \
  hera_bench_he_functions hera_bench_mpc_functions \
  hera_bench_conversion -j 2
ctest --test-dir build/full -R 'hera_bench_.*_smoke' --output-on-failure
```

QK、Attention×V（计时不含密钥生成和输入加密）：

```bash
./build/full/hera_bench_qk \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2
./build/full/hera_bench_attention_v \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2
```

其他 HE、双进程 MPC 和转换测试：

```bash
./build/full/hera_bench_he_functions --kernel=qkv --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=wo --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=wu --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=wd --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=softmax-exp --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=gelu-poly --warmup=1 --iterations=10

./build/full/hera_bench_mpc_functions --kernel=softmax-max \
  --tokens=2 --heads=1 --warmup=1 --iterations=10 --port=24900
./build/full/hera_bench_mpc_functions --kernel=softmax-normalize \
  --tokens=2 --heads=1 --warmup=1 --iterations=10 --port=24910
./build/full/hera_bench_mpc_functions --kernel=gelu-select \
  --tokens=2 --width=8 --warmup=1 --iterations=10 --port=24920

./build/full/hera_bench_conversion \
  --party=local --config=configs/conversion_smoke.json \
  --profile=conversion_test --port=24930 --threads=1 \
  --warmup=1 --iterations=10
```

自定义维度示例：

```bash
./build/full/hera_bench_he_functions \
  --kernel=qkv --tokens=4 --hidden=8 --heads=2 --kv-heads=1 \
  --qk-width=4 --value-width=4 --ffn-width=12 --degree=8192 \
  --he-threads=2 --warmup=1 --iterations=10
./build/full/hera_bench_he_functions --kernel=gelu-poly \
  --tokens=4 --ffn-width=12 --degree=8192 --warmup=1 --iterations=10
./build/full/hera_bench_mpc_functions --kernel=softmax-max \
  --tokens=8 --heads=2 --warmup=1 --iterations=10 --port=24940
```

`--tokens`：序列长度；`--heads`：query head 数；`--width`：QK 的 Q/K
每头宽度、Attention×V 的 V 每头宽度，或 MPC GeLU 宽度；`--degree`：CKKS
多项式次数；`--warmup`：不计时次数；`--iterations`：计时次数；
`--he-threads`：HE 线程数；转换的 `--threads`：协议 worker 数。
HE 投影还支持 `--hidden`、`--kv-heads`、`--qk-width`、`--value-width`、
`--ffn-width`。转换 benchmark 固定为每轮 8 个逻辑复数值，不支持批量维度扫描。
计时边界见 [QK/Attention×V](docs/benchmarks/he_qk_attention_v.md)和
[其他模块](docs/benchmarks/functional_kernels.md)。

## 配置与其他说明

- [单层 BERT-base 配置](configs/bert_base_single_layer_seq128.json)；
  [12 层可变宽度配置](configs/bert_base_variable_dims_seq128.json)。运行前先用
  `--dry-run` 检查资源。
- 仅测时间且内存不足时，可使用 `--weight-mode=virtual` 和
  `--weight-template-count=1` 复用合成 plaintext。该模式不保持模型数值语义，
  不能用于精度评估。
- online 计时不含 `Prepare()` 权重编码。`--he-threads`、
  `--conversion-workers`、`--he-memory-mib` 分别控制 HE 并行、转换 worker
  和 HE 临时任务内存预算。

协议、level/packing 与运行限制见 [实现说明](docs/IMPLEMENTATION.md)和
[复现说明](docs/REPRODUCIBILITY.md)；第三方来源及许可证注意事项见
[THIRD_PARTY.md](THIRD_PARTY.md)。
