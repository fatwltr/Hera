# HERA

HERA 是一个研究用途的混合 CKKS/MPC 私有 Transformer 推理原型。仓库提供配置驱动的计算图、复数 CKKS↔MPC 转换、五个 HE/MPC 阶段以及多层 Pre-LN Transformer 的双进程运行时。

公开边界是密文到密文：

```text
CKKS hidden states
  -> LN1 / QKV / QK
  -> Softmax
  -> Attention x V / Wo / LN2
  -> Wu / GeLU
  -> Wd / residual
  -> CKKS hidden states
```

中间执行路径不提供 decrypt 接口；`--verify` 只允许本机测试模式在最终输出处解密。

## 当前状态

- 支持多层 Pre-LN + GeLU、MHA/GQA/MQA 和非方阵 Wq/Wk/Wv/Wo/Wu/Wd。
- HE 参数由 JSON 中每个 region 的 profile 独立配置。
- 服务端持有模型与密文，客户端持有 CKKS secret key。
- MPC 使用修补后的 HERA-SCI VOLE/Silent-OT（Ferret/PCG）后端；SEAL 固定为 4.4.5。
- Softmax、LayerNorm rsqrt、GeLU 顺序复用同一个预热 MPC 会话；stage 完成后立即释放失效的密文/分享 payload。
- `runtime_tiny.json` 用于可执行回归；`figure10_smoke.json` 是资源开销较大的研究配置。
- 参数允许 `security_level=none`，因此本仓库不是生产安全实现。

详细设计和各阶段 level/packing contract 见 [实现说明](docs/IMPLEMENTATION.md)，复现边界与已知限制见 [复现说明](docs/REPRODUCIBILITY.md)。
本机 tiny profile 的基线记录位于
[`docs/benchmarks/`](docs/benchmarks/runtime_tiny_local_2026-09-19.md)。
从已打包密文开始的独立 QK、Attention×V HE benchmark 及与 BLB 源码的
对照边界见 [HE kernel benchmark](docs/benchmarks/he_qk_attention_v.md)。

## 环境

推荐 Ubuntu 22.04/24.04、x86-64、CMake 3.21+ 和支持 C++17 的编译器。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync \
  libssl-dev libgmp-dev libmpfr-dev libboost-dev
```

获取源码（仓库若为私有，先在服务器配置有权限的 GitHub SSH key）：

```bash
git clone git@github.com:fatwltr/Hera.git
cd Hera
git rev-parse --short HEAD
```

如果已配置 HTTPS 凭据，也可以改用 HTTPS 克隆。服务器上必须先拉取包含当前
源码改动的提交；仅将改动保留在开发机未提交的工作区，服务器克隆不到。

完整构建会从 `dependencies.lock` 指定的提交拉取：

- Microsoft SEAL 4.4.5；
- BLB 的 SCI 基线；
- BOLT 的非线性 SCI 实现。

依赖只写入仓库的 `.deps/`，不会安装到系统目录。
首次执行 `bootstrap_dependencies.sh` 需要访问 GitHub，以及 BOLT 的 Eigen
子模块所在的 GitLab；完成安装后可在同一目录离线重复执行。若服务器无法访问
这些地址，应先解决依赖下载或在目标服务器的共享文件系统上完成准备，不能只
复制开发机生成的含绝对路径的 `env.sh`。

## 快速验证框架

这个路径不需要 SEAL 或 SCI：

```bash
./scripts/build_and_test.sh framework 4
./build/framework/hera_private_inference \
  --config=configs/figure10_smoke.json \
  --dry-run --dump-graph
```

也可以使用 CMake preset：

```bash
cmake --preset framework
cmake --build --preset framework
ctest --preset framework
```

## 从干净 clone 复现完整运行时

```bash
./scripts/bootstrap_dependencies.sh .deps 4
./scripts/build_and_test.sh full 4 .deps

./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json \
  --party=local --verify --port=23100
```

这里的 `full` 构建脚本会运行全部 CTest，随后上面的命令才进行一次完整双进程
推理。成功时应看到 `HERA_BUILD_STATUS=PASS` 和
`HERA_PRIVATE_INFERENCE_STATUS=PASS`。请从仓库根目录运行；`--party=local`
在同一台服务器启动客户端与服务端，不需要第二台机器。端口 23100 起的协议端口
不能被其他进程占用。已检出正确依赖提交且构建完整时，重新执行依赖准备脚本
不再需要联网。

若只想先检查配置、而不启动密码协议，可执行：

```bash
./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json --dry-run
```

`configs/runtime_tiny.json` 是首次运行建议使用的配置；128-token 或 12 层
BERT 配置需要远高于 tiny 的内存与运行时间，不应作为安装后的第一条验证命令。

仅测量密码运算开销、且机器无法容纳全部编码权重时，可以启用 benchmark-only
虚拟权重池：

```bash
./build/full/hera_private_inference \
  --config=configs/bert_base_variable_dims_seq128.json \
  --party=local --port=23100 \
  --he-threads=8 --conversion-workers=4 --he-memory-mib=1024 \
  --weight-mode=virtual --weight-template-count=1
```

该模式仍执行原定数量的 rotation、`multiply_plain`、累加、rescale、
conversion 和 MPC 操作，但兼容的 Wq/Wk/Wv/Wo/Wu/Wd diagonals 会重复使用
有限数量的物理 plaintext。它用于低内存性能测试，不保持模型数值语义，也不能
作为模型精度结果。使用更大的 `--weight-template-count` 可以部分恢复真实的缓存/
内存访问压力。正常模式为 `--weight-mode=materialized`。

模型权重的 CKKS plaintext 编码不能在真实推理中直接省略：矩阵对角线和
LayerNorm 参数必须先变成与具体 HE profile、level、scale 和 packing 匹配的
plaintext。运行时在模型初始化时调用一次 `Prepare()`，随后冻结缓存；同一个
runtime 的 warmup 和 `--repeat` 推理均复用这些结果，online 计时不包括该步骤。
准备阶段只构造携带 `parms_id`、scale 和 packing 的元数据 tensor，直接编译
LayerNorm 与各投影的 plaintext；不会为了编译权重而加密零输入或执行 HE
乘法、旋转。实际密文计算只在推理阶段发生。
`Prepare()` 还会把 Wq/Wk/Wv/Wo/Wu/Wd 的 slot mask、旋转分组和编码后的
diagonal 编译成不可变投影计划。在线投影只校验输入 contract 并执行计划，不再
逐项重建 mask 或哈希整条 slot 向量。报告中的
`HERA_PREPARED_PROJECTION_PLANS` 是计划数，
`HERA_ONLINE_PROJECTION_PLAN_HITS` 是在线调用累计命中数；
`--repeat=2` 时后者应为单次调用数的两倍。
模型矩阵在注册后以只读共享对象保存，在线阶段的 Wq/Wk/Wv/Wo/Wu/Wd
读取不再复制整张矩阵；Wo 命中计划时也直接复用准备阶段的输出 packing。
报告中的 `HERA_PREPARE_MS` 属于 setup，`HERA_ONLINE_MODEL_ENCODES=0`
表示 online 没有首次编码模型参数。每次重新启动程序仍需初始化一次；若希望
跨进程跳过它，需要另行设计与 SEAL 版本及参数配置绑定的持久化缓存。

`--he-threads` 和 `--conversion-workers` 分别设置 HE 与转换协议的线程预算；
`--threads` 仍可作为同时设置两者的兼容写法。HE 侧
Block 2 并行 V 与 Q/K 主干，QK 的独立 diagonal 输出组最多再并行 2 路；
Attention×V 的独立 value 输出组最多并行 2 路；Softmax 多项式最多并行
4 个密文，GeLU 多项式最多并行 2 个密文；Wo/Wu/Wd 的独立输出密文组
最多并行 2 路。HE 任务共享可复用线程池，只有一个输出组或密文时不会
提交并行任务。`--he-memory-mib`（默认 512）按预估每组活跃密文内存限制
并行组数；这是临时任务预算，不是整个程序的峰值内存上限。增大
`--conversion-workers` 会增加离线 OT 初始化成本，建议按目标配置测量。
运行报告中的 `HERA_HE_POOL_WORKERS` 显示实际创建的复用工作线程数。
conversion 会按多项式次数、FFT 精度、工作位宽和参与方复用固定点
CKKS 变换表。多密文批次在 `--conversion-workers` 大于 1 时，最多同时运行 2 个
纯本地变换任务；OT 和协议网络帧仍按密文顺序执行，不会并发调用同一会话。
运行报告中的 `HERA_CONVERSION_TRANSFORM_CACHE_HITS/MISSES` 和
`HERA_CONVERSION_LOCAL_PEAK_INFLIGHT` 可用于确认是否实际命中这项优化。

SEAL 密文经 256 KiB 缓冲区分块收发，不再通过完整 `stringstream` 和
`string` 复制；公钥包也分块发送，密钥序列化直接写入最终字节数组。
局域网默认 `--seal-compression=none`；跨机带宽受限时可以尝试
`--seal-compression=zstd`，但必须以 `HERA_SEAL_COMPRESSION=zstd`
重新构建依赖，并让两端使用相同构建与参数。当前默认构建未启用压缩，
请求不可用的压缩模式会在启动时明确报错。是否值得压缩应以目标网络的
传输时间与压缩 CPU 开销实测决定。

如需构建 zstd 版本，可在独立依赖目录执行：

```bash
HERA_SEAL_COMPRESSION=zstd ./scripts/bootstrap_dependencies.sh .deps-zstd 4
./scripts/build_and_test.sh full 4 .deps-zstd
```

等价的一键入口是：

```bash
./scripts/reproduce.sh 4
```

完整测试包含真实的两方 conversion、MPC 非线性、Block 2、Softmax、Attention×V/LN2、FFN/GeLU、Block 5 和整体运行时。它明显慢于 framework-only 测试。

## QK 与 Attention×V 单算子 benchmark

安装上文列出的系统依赖后，以下命令从克隆仓库开始，依次准备依赖、配置、
仅构建两个 benchmark、运行冒烟测试和计时。若已经位于仓库根目录，跳过前两行；
若依赖已准备好，也可跳过 `bootstrap_dependencies.sh`。

```bash
git clone git@github.com:fatwltr/Hera.git
cd Hera
./scripts/bootstrap_dependencies.sh .deps 4
cmake --preset full
cmake --build build/full --target hera_bench_qk hera_bench_attention_v -j 2
ctest --test-dir build/full -R 'hera_bench_(qk|attention_v)_smoke' --output-on-failure

./build/full/hera_bench_qk \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2

./build/full/hera_bench_attention_v \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2
```

`bootstrap_dependencies.sh` 的 `.deps` 是依赖安装目录，`4` 是构建并行数；
`cmake --build` 的 `-j 2` 是编译并行数。运行参数中，`--tokens` 是序列长度，
`--heads` 是 attention head 数，`--width` 在 QK 测试中是 Q/K head width，
在 Attention×V 测试中是 V head width；`--degree` 是 CKKS 多项式次数，
`--warmup` 是不计时的预热次数，`--iterations` 是计时次数，`--he-threads`
是 HE 计算线程数。

计时从已按各 kernel 要求打包的输入密文开始，包含对应 HE kernel，
不包括 context/key 生成、输入编码与加密、最终解密、conversion、Softmax 或网络。
两个 benchmark 使用仅满足各自计算的研究用模数链，并非完整 Transformer 的
参数。可选参数 `--blb-ms=<毫秒>` 用于输入同形状、同硬件、同线程数、同计时
边界下的 BLB 实测均值，输出时间比；未提供时不会臆测 BLB 基线。具体参数、输出
解释和可比性限制见 [benchmark 说明](docs/benchmarks/he_qk_attention_v.md)。

## 其他功能模块 benchmark

以下命令在仓库根目录执行，包含依赖准备、配置、仅构建新增 benchmark、冒烟测试，
以及每个功能的独立计时；若已完成上面的依赖准备和配置，可跳过前两行。

```bash
./scripts/bootstrap_dependencies.sh .deps 4
cmake --preset full
cmake --build build/full --target \
  hera_bench_he_functions hera_bench_mpc_functions hera_bench_conversion -j 2
ctest --test-dir build/full -R 'hera_bench_(he|mpc|conversion).*_smoke' \
  --output-on-failure

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

`--kernel` 选择测试的函数；`softmax-exp` 和 `gelu-poly` 只测 HE 多项式，
`softmax-max`、`softmax-normalize`、`gelu-select` 只测 MPC 部分。
`--warmup` 是不计时的预热次数，`--iterations` 是计时次数；`--port` 是本机
双进程协议端口，运行前需确保端口空闲。HE 测试还可设置 `--tokens`、
`--hidden`、`--heads`、`--kv-heads`、`--qk-width`、`--value-width`、
`--ffn-width`、`--degree` 和 `--he-threads`；conversion 的 `--threads` 是
协议 worker 数。默认 HE 测试尺寸刻意设得很小，适合先验证构建。详细计时边界和
限制见 [功能模块 benchmark 说明](docs/benchmarks/functional_kernels.md)。

## 12 层 BERT-base 可变内部宽度配置

> 仅测试性能

[`configs/bert_base_variable_dims_seq128.json`](configs/bert_base_variable_dims_seq128.json) 提供以下结构：

```text
tokens=128, model_width=768, q_heads=12, kv_heads=12
per-layer [qk_head_width, value_head_width, ffn_width]
[48,64,3072], [48,32,3056], [48,64,3072], [64,64,3056]
[64,64,3072], [64,64,2768], [64,48,3072], [48,64,2464]
[48,64,2144], [48,64,1232], [48,48,1232], [32,48,624]
```

这里的 QK/VO 宽度均为单个 head 的宽度。Wq/Wk/Wv/Wo/Wu/Wd 的非方阵尺寸由配置自动推导。


## 仓库结构

```text
app/                    CLI 入口，不包含可复用协议逻辑
cmake/                  framework/crypto 构建定义和公共 helper
configs/                tiny、conversion 和研究配置
include/hera/           稳定的公共 C++ 接口
src/framework/          配置、类型化图、模块 contract、recording backend
src/crypto/conversion/  复数 CKKS↔MPC 协议
src/crypto/he/          五个融合 HE/MPC stage executor
src/crypto/runtime/     整体编排、协调、abort/heartbeat
tests/                  单元、协议和端到端测试
scripts/                依赖准备、构建和复现入口
third_party/            可复现的 HERA-SCI 补丁
```

`hera_framework` 完全不依赖 SEAL/SCI；真实密码实现只存在于
`src/crypto`。通常只使用 `framework` 或 `full` 两个 preset。细粒度的
`HERA_ENABLE_*` 开关仅供单独开发某个 kernel 使用。
