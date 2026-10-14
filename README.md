# HERA

HERA 是 CKKS/MPC 私有 Transformer 推理原型，支持多层 Pre-LN、图 10 的归一化输出残差模式及 GeLU、
MHA/GQA/MQA 和可变宽度。输入、输出均为 CKKS 密文。当前配置允许
`security_level=none`，**仅供研究，不能用于生产安全部署**。

Wu、Wo、Wd 均采用 complex→real 投影。Wu 直接生成 GeLU 使用的实数列，
LN2/Wu/GeLU 固定消耗五层；示例 FFN 链为 `[60,60,46,46,46,60]`。
原 `ffn_real_output` 开关和独立 Wu realify 路径已移除，相关实现位于 `src/crypto/he/`。

## 在 Ubuntu 服务器上构建并验证

Ubuntu 22.04/24.04、x86-64；需访问 GitHub/GitLab，并具备仓库克隆权限：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync ca-certificates python3 util-linux \
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
默认启用 Zstd 压缩序列化，密文通过 256 KiB 缓冲区分块传输；运行时可用
`--seal-compression=none` 关闭压缩。已有未启用压缩的依赖目录不会被 bootstrap
覆盖，需要在新目录重新构建依赖，或显式使用 `HERA_SEAL_COMPRESSION=none`。
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

## 在新机器复现两个 BERT-base 单层测试

使用包含当前改动的同一份项目源码，尤其需要下列两个 JSON、`dependencies.lock`、
`third_party/hera-sci.patch` 和 `scripts/seal_rotation_hoisting.patch`。
若从远端克隆，选择已经包含这些文件和改动的分支或提交。在新机器重新构建依赖和主程序；
构建目录含绝对路径及 CPU 相关编译选项。

| 测试 | 配置 | tokens / 隐藏宽度 / heads | QK每头宽度 | V每头宽度 | FFN宽度 |
| --- | --- | --- | ---: | ---: | ---: |
| 普通单层 | [bert_base_single_layer_seq128.json](configs/bert_base_single_layer_seq128.json) | 128 / 768 / 12 | 64 | 64 | 3072 |
| 压缩单层 | [bert_base_single_layer_seq128_reduced_layer11.json](configs/bert_base_single_layer_seq128_reduced_layer11.json) | 128 / 768 / 12 | 32 | 48 | 624 |

压缩配置从12层可变维度配置提取 `layer11`；两份配置使用相同的普通紧凑链及 MPC43/scale13、κ40。
以下是单机双进程、虚拟权重的开销测试。建议至少32 GiB RAM；本机6 workers时双方进程树
RSS采样峰值约17.50/15.93 GiB，此外还需为操作系统和其他程序留出内存。

安装上一节的系统依赖后，在项目根目录使用 Bash 执行：

```bash
set -euo pipefail
test -f configs/bert_base_single_layer_seq128.json
test -f configs/bert_base_single_layer_seq128_reduced_layer11.json

./scripts/bootstrap_dependencies.sh .deps 2
OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_MAX_ACTIVE_LEVELS=1 \
  ./scripts/build_and_test.sh full 2 .deps

./build/full/hera_private_inference \
  --config=configs/bert_base_single_layer_seq128.json --dry-run
./build/full/hera_private_inference \
  --config=configs/bert_base_single_layer_seq128_reduced_layer11.json --dry-run
```

bootstrap固定构建 SEAL 4.4.5（含旋转优化和Zstd）及当前HERA-SCI ABI3/Ferret依赖，
其中包含 `MPC_COMPARISON=CORRELATED_AND_V1` 标记；build脚本自动读取 `.deps/env.sh`，
以Release构建并串行运行回归。编译参数 `2` 只控制并行编译任务数。
成功标志依次为 `HERA_DEPENDENCIES_STATUS=PASS`、`HERA_BUILD_STATUS=PASS`、
两次 `HERA_DRY_RUN_STATUS=PASS`。dry-run只检查配置和计算图，不执行真实HE/MPC，也不验证内存是否足够。

先查看新机器的CPU拓扑和当前进程允许使用的CPU：

```bash
lscpu -e=CPU,CORE,SOCKET,ONLINE
taskset -pc $$
free -h
```

从允许的CPU中选6个不同物理核，每个核选一个逻辑CPU，填写下面的 `HERA_BENCH_CPUS`。
示例 `0,2,4,6,8,10` 来自本机i5-14400F，其他机器应按自己的拓扑修改。
两项测试使用相同的CPU集合和worker数，按顺序执行：

```bash
HERA_BENCH_CPUS=0,2,4,6,8,10
HERA_BENCH_WORKERS=6
HERA_BENCH_OUTPUT="build/benchmarks/bert_base_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$HERA_BENCH_OUTPUT"
taskset -c "$HERA_BENCH_CPUS" true
lscpu > "$HERA_BENCH_OUTPUT/cpu.txt"
git rev-parse HEAD > "$HERA_BENCH_OUTPUT/commit.txt"

run_bert_base() {
  local case_name=$1 config_path=$2 base_port=$3
  OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 OMP_MAX_ACTIVE_LEVELS=1 \
  taskset -c "$HERA_BENCH_CPUS" ./build/full/hera_private_inference \
    --config="$config_path" --party=local --port="$base_port" \
    --threads="$HERA_BENCH_WORKERS" \
    --weight-mode=virtual --weight-template-count=1 \
    --he-memory-mib=1024 --plaintext-cache-mib=1024 \
    --conversion-batch-elements=0 --projection-baby-steps=0 \
    --seal-compression=zstd --stage-timeout-seconds=600 \
    --warmup=1 --repeat=3 --verify \
    2>&1 | tee "$HERA_BENCH_OUTPUT/$case_name.log"
  test "$(grep -c '^HERA_PRIVATE_INFERENCE_STATUS=PASS$' \
    "$HERA_BENCH_OUTPUT/$case_name.log")" -eq 2
}

run_bert_base standard configs/bert_base_single_layer_seq128.json 27900
run_bert_base reduced configs/bert_base_single_layer_seq128_reduced_layer11.json 28800
```

`--party=local`自动启动双方，无需再手工启动客户端或服务器；每个日志应有两个PASS。
保证两次测试预留的端口区间 `27900–28700`、`28800–29600` 空闲。
`--threads=6`设置HE、MPC和转换各6 workers，OpenMP2控制SCI内部协议并行，
不会将外层worker限制为2。若核数或内存不足，两个测试一起降低worker数并调整CPU列表；
`--he-memory-mib`和`--plaintext-cache-mib`仅限制对应缓存/临时任务，不是总进程内存上限。

在同一个Bash会话汇总结果：

```bash
python3 - "$HERA_BENCH_OUTPUT" <<'PY'
from pathlib import Path
import json
import re
import sys

directory = Path(sys.argv[1])
results = {}
for name in ("standard", "reduced"):
    text = (directory / f"{name}.log").read_text()
    assert text.count("HERA_PRIVATE_INFERENCE_STATUS=PASS") == 2, name
    def values(key):
        found = re.findall(r"^" + re.escape(key) + r"=([0-9.eE+-]+)$", text, re.M)
        assert len(found) == 2, (name, key, found)
        return [float(value) for value in found]
    results[name] = {
        "online_mean_s": max(values("HERA_END_TO_END_ONLINE_MS")) / 1000,
        "setup_s": max(values("HERA_SETUP_MS")) / 1000,
        "last_formal_compute_send_MB": sum(values("HERA_COMMUNICATION_BYTES")) / 1e6,
    }
for key in results["standard"]:
    a, b = results["standard"][key], results["reduced"][key]
    print(f"{key}: standard={a:.3f}, reduced={b:.3f}, reduction={(1-b/a)*100:.2f}%")
(directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
PY
```

`HERA_END_TO_END_ONLINE_MS`是3次正式推理的平均值，包含HE、MPC、转换和最终输出传输；
排除初始化、权重Prepare、1次预热及最后的解密检查。双方时间取较大者，不相加。
`HERA_SETUP_MS`已包含Prepare，单独记录。`HERA_COMMUNICATION_BYTES`仅代表最后一次正式推理，
汇总双方发送量，不包含初始化密钥/输入及阶段外最终输出传输；MB按10^6 B。
虚拟权重测试用于测量计算几何的开销，`--verify`检查最终解密输出有限性，不验证预训练模型精度。

本机上述条件下的参考结果为普通层 **50.40秒、728.67 MB**，压缩层 **30.12秒、425.82 MB**；
新机器应以自己的日志为准，测试原始日志保存在上面的本地输出目录。

## 单算子 benchmark

只构建和测试 benchmark：

```bash
source .deps/env.sh
cmake -S . -B build/full -DCMAKE_BUILD_TYPE=Release \
  -DHERA_ENABLE_FULL_RUNTIME=ON \
  -DSEAL_DIR="$SEAL_DIR" -DSCI_DIR="$SCI_DIR" -DEigen3_DIR="$Eigen3_DIR"
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
`--ffn-width`。转换 benchmark 支持 `--ciphertexts=N`；默认每密文使用8个逻辑实数，
`--dense` 使用所有实、虚槽，并逐轮检查 C2M、检查最后一轮往返误差。
`--batch-elements=N` 按系数数量限制安全转换批次，0（默认）按每个worker 16384系数选择；
至少处理一个完整密文。较大批次会增加内存，未必更快。
单算子benchmark测量对应内核或转换协议的开销，不代表完整Transformer层的在线耗时。

## 配置与其他说明

- [单层 BERT-base 配置](configs/bert_base_single_layer_seq128.json)；
  [压缩 BERT-base 单层配置](configs/bert_base_single_layer_seq128_reduced_layer11.json)；
  [单层 BERT-large 尺寸配置](configs/bert_large_single_layer_seq128.json)；
  [12 层可变宽度配置](configs/bert_base_variable_dims_seq128.json)。运行前先用
  `--dry-run` 检查配置和计算图。
- 仅测时间且内存不足时，可使用 `--weight-mode=virtual` 和
  `--weight-template-count=1` 复用合成 plaintext。该模式不保持模型数值语义，
  不能用于精度评估。
- online 计时不含 `Prepare()` 权重编码。`--he-threads`、
  `--conversion-workers`、`--mpc-workers` 分别控制 HE、转换和非线性 MPC 并行。
  MPC 默认跟随 HE 线程数（最多50个不冲突的 SCI socket lane）；设为1可回到单会话调度。
  GeLU按元素、Softmax按完整行划分，独立OT会话在Setup准备并跨阶段复用。
  `--he-memory-mib` 只控制 HE 临时任务内存，不包含 MPC 的 OT 状态和批次内存。
  两方必须使用相同的MPC worker数及新版本程序，运行时会校验。
- 转换支持 `--conversion-batch-elements=N`（0为自适应），两方需使用相同的实际批次和worker数。
  当前转换协议为v14，双方必须同时更新；依赖需包含
  `OT_EXPANSION=UNROLLED_AES_PREFIX_PAD`、`MPC_MEMORY=REUSABLE_SLAB_V1`、
  `MPC_COMPARISON=CORRELATED_AND_V1`，bootstrap和CMake会校验。
- 实数转换边界现统一双路打包，在后续算法需要时恢复实数布局；拆分的1/2吸收到转换编码或已有明文中，
  不增加rescale或HE层数。比较查表复用连续内存，按公开布局处理广播/填充，
  通用进位使用相关AND三元组和radix4，相关实现位于 `src/crypto/conversion/`。
- 投影 NTT 权重使用有界缓存，`--plaintext-cache-mib=256` 为默认驻留预算；
  超出部分写入临时文件，在线按需加载。此预算不包含公开掩码、密钥和任务中的
  临时数据。`--projection-baby-steps=0` 使用默认成本估计，也可指定候选步长。
- 大规模稠密权重的单次投影可能超过默认30秒阶段等待时间；可用
  `--stage-timeout-seconds=600` 延长阶段同步期限，心跳失联期限仍为30秒。

## Git提交范围

`.gitignore`仅排除明确的生成产物和本地调试文件。核心源码与头文件、CLI、CMake/CI、
正式配置、依赖锁文件和必要补丁、构建/复现脚本、回归测试及README均正常提交；
新增源码、配置、脚本或测试无需修改忽略规则。

整个`docs/`目录、下载的第三方源码、编译产物、`build/`下的测量日志和原始数据、Python缓存，
以及明确列出的BLB对照驱动和临时benchmark脚本
留在本地。测试输出目录应使用
`build/benchmarks/`；这些文件不会因普通的`git add`进入后续提交。
已在Git历史中的文件不受新忽略规则追溯删除；已跟踪的调试文件需要从索引移除后才会被忽略。

协议及level/packing接口位于 `include/hera/`，实现位于 `src/crypto/`；
第三方来源及许可证注意事项见
[THIRD_PARTY.md](THIRD_PARTY.md)。
