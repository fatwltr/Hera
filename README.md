# HERA

基于 CKKS/MPC 的私有 Transformer 推理原型，支持单层、多层和可变维度配置。
示例参数仅用于研究。

## 云服务器构建与测试

使用 Linux x86-64 服务器（Ubuntu 22.04/24.04），CPU 需要 AES-NI、AVX2 和 SSE4.1；不需要 GPU 或 RDSEED。默认回归建议至少 2 vCPU、8 GiB 内存、20 GiB 可用磁盘。首次下载需要访问 GitHub/GitLab，依赖从固定提交构建，不需要机器上已有 SEAL、SCI 或 EMP。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync ca-certificates python3 util-linux \
  libssl-dev libgmp-dev libmpfr-dev libboost-dev libzstd-dev zlib1g-dev

git clone https://github.com/fatwltr/Hera.git
cd Hera

./scripts/bootstrap_dependencies.sh .deps 2
./scripts/build_and_test.sh full 2 .deps
```

`2` 是编译并行数。脚本先检查工具、CMake（至少 3.21）和 CPU 指令集，再下载源码、安装依赖、以 Release 编译并顺序运行全部 19 项 CTest 回归。覆盖转换、HE/MPC 算子、双层推理及 OpenMP 单线程限制下的端到端正确性，不跳过完整集成测试。

成功时应看到：

```text
HERA_DEPENDENCIES_STATUS=PASS
100% tests passed, 0 tests failed out of 19
HERA_BUILD_STATUS=PASS
```

也可用 `./scripts/reproduce.sh 2` 一次执行上述依赖构建、项目构建和测试。
依赖位于 `.deps/`，程序位于 `build/full/`。准备脚本只修改生成的依赖副本，重复运行相同命令会复用已完成的依赖；换机器后需重新编译。

仅下载固定版本源码、暂不编译：

```bash
./scripts/download_dependencies.sh .deps
```

下载成功输出 `HERA_DEPENDENCY_DOWNLOAD_STATUS=PASS`。之后运行前面的 bootstrap 和 build/test 命令即可。压缩使用系统 `libzstd-dev`，构建过程无需再下载 Microsoft GSL 或 Zstandard 源码。

编译并行数与运行时 workers 独立。构建脚本默认把 OpenMP 限制为 2 个线程，CTest 也设置相同的可复现环境；修补后的 SCI 在 `OMP_THREAD_LIMIT=1` 或动态缩小线程团队时仍执行全部通信方向。

## 小规模运行检查

完整测试通过后，运行一个物化权重的双进程 tiny 推理：

```bash
./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json --party=local --port=27100 \
  --threads=2 --verify
```

应输出两个 `HERA_PRIVATE_INFERENCE_STATUS=PASS`。`--party=local` 自动启动双方，无需两台机器；保持本机端口区间 `27100–27600` 空闲。

只检查较大配置和计算图：

```bash
./build/full/hera_private_inference \
  --config=configs/bert_large_single_layer_seq128.json --dry-run
```

## BERT-large 单层测试

使用 [BERT-large 单层配置](configs/bert_large_single_layer_seq128.json)：
128 tokens、隐藏宽度1024、16 heads、QK/V每头64维、FFN宽度4096，使用普通紧凑链。

6 workers建议至少32 GiB内存。

```bash
set -o pipefail
export OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_MAX_ACTIVE_LEVELS=1
mkdir -p build/benchmarks

./build/full/hera_private_inference \
  --config=configs/bert_large_single_layer_seq128.json \
  --party=local --port=27900 --threads=6 \
  --weight-mode=virtual --weight-template-count=1 \
  --he-memory-mib=1024 \
  --stage-timeout-seconds=600 --warmup=1 --repeat=3 --verify \
  2>&1 | tee build/benchmarks/bert_large_single_layer.log
```

`--party=local`自动启动双方，应输出两个 `HERA_PRIVATE_INFERENCE_STATUS=PASS`。
需保证端口区间 `27900–28700` 空闲。

## 查看结果

Transformer block 测试仅输出双方标识、每个 block 的六项在线指标和最终状态；多层模型另输出 `HERA_TOTAL` 汇总。

| 指标 | 含义 |
| --- | --- |
| `HE_MS` | HE 算子的执行时间 |
| `MPC_MS` | Softmax、LayerNorm 倒平方根、GeLU 等非线性 MPC 的执行时间 |
| `CONVERSION_MS` | HE/MPC 转换时间，包含转换内部的加解密、FFT、OT 与密文传输 |
| `HE_COMM_BYTES` | 本地 HE 算子通信量，当前为 0 |
| `MPC_COMM_BYTES` | 非线性 MPC 发送的字节数 |
| `CONVERSION_COMM_BYTES` | 转换协议及其密文传输发送的字节数 |

每个 block 输出一行 `HERA_BLOCK=layer0,HE_MS=...,MPC_MS=...,CONVERSION_MS=...,HE_COMM_BYTES=0,MPC_COMM_BYTES=...,CONVERSION_COMM_BYTES=...`。
时间单位为毫秒，时间和通信量均为正式轮次的每次平均值。`--warmup=1 --repeat=3` 表示预热一次、正式测量三次。


## 单独测试 QK、XW、Attention×V

完整构建会生成 `build/full/hera_he_benchmark`，不加入默认回归测试。
它读取 JSON 中指定层的维度和 HE profile，使用合成输入、权重，直接调用生产算子。

```bash
cfg=configs/bert_large_single_layer_seq128.json
./build/full/hera_he_benchmark --op=qk --config="$cfg" --threads=6 --he-memory-mib=1024 --warmup=1 --repeat=3
./build/full/hera_he_benchmark --op=xw --projection=wq --config="$cfg" --threads=6 --he-memory-mib=1024 --warmup=1 --repeat=3
./build/full/hera_he_benchmark --op=av --config="$cfg" --threads=6 --he-memory-mib=1024 --warmup=1 --repeat=3
```

XW 可选择 `wq/wk/wv/wo/wu/wd`，每次测一个投影，包含其生产实现中的打包、缩放和偏置。
单独测 Q/K/V 时各自包含 baby rotations；三个结果相加不会体现联合 QKV 的旋转复用。
默认使用实际合成权重；大矩阵可加 `--weight-mode=virtual` 降低预处理和存储成本，
此模式跳过明文数值对照。实际权重模式最多抽查256个输出，小规模输出全部检查。

矩阵乘法测试输出：

```text
HERA_BENCHMARK=qk
HERA_KS_PER_RUN=实际调用次数
HERA_MEAN_MS=正式执行平均毫秒数
HERA_BENCHMARK_STATUS=PASS
```


第三方来源及许可证说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
