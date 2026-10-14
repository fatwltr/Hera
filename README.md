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

先运行小规模测试，检查环境：

```bash
./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json --party=local --verify --port=23100
```

## BERT-base 单层测试

两项测试均为128 tokens、隐藏宽度768、12 heads，使用普通紧凑链。
压缩层从可变维度配置的 `layer11` 提取。

| 配置 | QK每头宽度 | V每头宽度 | FFN宽度 |
| --- | ---: | ---: | ---: |
| [普通层](configs/bert_base_single_layer_seq128.json) | 64 | 64 | 3072 |
| [压缩层](configs/bert_base_single_layer_seq128_reduced_layer11.json) | 32 | 48 | 624 |

在同一个 Bash 会话中执行以下命令，两项测试依次运行。6 workers建议至少32 GiB内存。

```bash
set -o pipefail
export OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_MAX_ACTIVE_LEVELS=1
mkdir -p build/benchmarks

# 普通层
./build/full/hera_private_inference \
  --config=configs/bert_base_single_layer_seq128.json \
  --party=local --port=27900 --threads=6 \
  --weight-mode=virtual --weight-template-count=1 \
  --he-memory-mib=1024 --plaintext-cache-mib=1024 \
  --stage-timeout-seconds=600 --warmup=1 --repeat=3 --verify \
  2>&1 | tee build/benchmarks/bert_base_standard.log

# 压缩层
./build/full/hera_private_inference \
  --config=configs/bert_base_single_layer_seq128_reduced_layer11.json \
  --party=local --port=28800 --threads=6 \
  --weight-mode=virtual --weight-template-count=1 \
  --he-memory-mib=1024 --plaintext-cache-mib=1024 \
  --stage-timeout-seconds=600 --warmup=1 --repeat=3 --verify \
  2>&1 | tee build/benchmarks/bert_base_reduced.log
```

`--party=local`自动启动双方，每项测试应输出两个 `HERA_PRIVATE_INFERENCE_STATUS=PASS`。
需保证端口区间 `27900–28700`、`28800–29600` 空闲。

## 查看结果

日志中的主要指标：

| 指标 | 含义 |
| --- | --- |
| `HERA_END_TO_END_ONLINE_MS` | 正式推理平均耗时，含HE、MPC、转换和最终输出传输；双方取较大值 |
| `HERA_SETUP_MS` | 初始化耗时，已包含权重Prepare；双方取较大值 |
| `HERA_COMMUNICATION_BYTES` | 最后一次正式推理的计算协议发送量；双方相加 |

`--warmup=1 --repeat=3`表示预热一次、正式测量三次，在线均值不包含初始化和预热。
通信量不包含初始化密钥/输入和阶段外最终输出传输。
本机i5-14400F绑定六个P-core时，普通层约 **50.40秒 / 728.67 MB**，
压缩层约 **30.12秒 / 425.82 MB**；其他机器以实际测量为准。

## 常用说明

- `--threads=6`设置HE、MPC、转换各6 workers；也可分别使用
  `--he-threads`、`--mpc-workers`、`--conversion-workers`。内存或核数不足时，两项测试一起降低worker数。
- 虚拟权重用于性能测试；`--verify`检查最终解密输出有限性，不代表预训练模型精度验证。
- `--dry-run`只检查配置和计算图，不执行实际HE/MPC计算。
- 其他配置：[BERT-large单层](configs/bert_large_single_layer_seq128.json)、
  [12层可变维度](configs/bert_base_variable_dims_seq128.json)。
- `.gitignore`排除`docs/`、依赖副本、构建产物、日志、缓存和指定的调试脚本。
  源码、配置、测试、必要补丁和构建脚本正常提交。

第三方来源及许可证说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
