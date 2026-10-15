# HERA

基于 CKKS/MPC 的私有 Transformer 推理原型，支持单层、多层和可变维度配置。
示例参数仅用于研究，部分 HE profile 使用 `security_level=none`。

## 构建

支持 Ubuntu 22.04/24.04、x86-64。首次构建需要访问 GitHub/GitLab。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync ca-certificates python3 util-linux \
  libssl-dev libgmp-dev libmpfr-dev libboost-dev

git clone git@github.com:fatwltr/Hera.git
cd Hera

./scripts/bootstrap_dependencies.sh .deps 2
./scripts/build_and_test.sh full 2 .deps
```

脚本会下载固定版本依赖、以 Release 编译并运行回归测试。
`2` 是编译并行数；依赖位于 `.deps/`，程序位于 `build/full/`。
构建成功后输出 `HERA_BUILD_STATUS=PASS`。新机器请重新编译依赖和程序。

构建会运行16项关键回归，覆盖配置、转换、HE/MPC算子及双层端到端正确性。

如果只想先下载依赖源码，不进行编译或安装，运行：

```bash
./scripts/download_dependencies.sh .deps
```

源码保存在 `.deps/src/`，下载成功后输出 `HERA_DEPENDENCY_DOWNLOAD_STATUS=PASS`。
之后运行 `./scripts/bootstrap_dependencies.sh .deps 2` 编译并安装依赖，
再运行 `./scripts/build_and_test.sh full 2 .deps` 编译并测试 Hera。

先检查 BERT-large 配置和计算图：

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
  --he-memory-mib=1024 --plaintext-cache-mib=1024 \
  --stage-timeout-seconds=600 --warmup=1 --repeat=3 --verify \
  2>&1 | tee build/benchmarks/bert_large_single_layer.log
```

`--party=local`自动启动双方，应输出两个 `HERA_PRIVATE_INFERENCE_STATUS=PASS`。
需保证端口区间 `27900–28700` 空闲。

## 查看结果

日志中的主要指标：

| 指标 | 含义 |
| --- | --- |
| `HERA_END_TO_END_ONLINE_MS` | 正式推理平均耗时，含HE、MPC、转换和最终输出传输；双方取较大值 |
| `HERA_SETUP_MS` | 初始化耗时，已包含权重Prepare；双方取较大值 |
| `HERA_COMMUNICATION_BYTES` | 最后一次正式推理的计算协议发送量；双方相加 |

`--warmup=1 --repeat=3`表示预热一次、正式测量三次，在线均值不包含初始化和预热。
通信量不包含初始化密钥/输入和阶段外最终输出传输。

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
此模式会明确跳过明文数值对照。实际权重模式最多抽查256个输出，小规模输出全部检查。

`HERA_MEAN_MS`、`HERA_MEDIAN_MS`、`HERA_MIN_MS` 为纯 HE 算子耗时，
不含密钥生成、输入加密、权重/掩码预处理、预热和解密校验；这些算子不包含 MPC 或域转换。
`HERA_SETUP_MS`、`HERA_PREPARE_MS`、`HERA_WARMUP_MS` 单独报告。
`HERA_KS_PER_RUN` 是单次执行的 **旋转 + 共轭 + 重线性化** 次数，三项分别输出，
`HERA_KS_MEASURED_TOTAL` 为正式重复的总次数，不含预热；`HERA_ROTATION_KEY_COUNT` 是生成的密钥数量。
输出还包含输入/输出 level、密文数和明文校验误差。参数说明见 `--help`。

## 常用说明

- `--threads=6`设置HE、MPC、转换各6 workers；也可分别使用
  `--he-threads`、`--mpc-workers`、`--conversion-workers`。内存或核数不足时降低worker数。
- worker数没有固定性能上限；HE按任务数和`--he-memory-mib`预算调度，
  LayerNorm和GeLU MPC按输入元素分片。阶段日志中的`LN_HE_WORKERS`、
  `LN_MPC_WORKERS`、`GELU_MPC_WORKERS`显示实际使用数量。
  较大worker数会扩展端口区间，MPC起始端口见`HERA_MPC_BASE_PORT`；
  仍需满足TCP端口范围，双方使用相同参数及同一版本程序。
- 虚拟权重用于性能测试；`--verify`检查最终解密输出有限性，不代表预训练模型精度验证。
- `--dry-run`只检查配置和计算图，不执行实际HE/MPC计算。
- `.gitignore`排除`docs/`、依赖副本、构建产物、日志、缓存和指定的调试脚本。
  源码、配置、测试、必要补丁和构建脚本正常提交。

第三方来源及许可证说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
