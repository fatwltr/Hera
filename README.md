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

## 环境

推荐 Ubuntu 22.04/24.04、x86-64、CMake 3.21+ 和支持 C++17 的编译器。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync \
  libssl-dev libgmp-dev libmpfr-dev
```

完整构建会从 `dependencies.lock` 指定的提交拉取：

- Microsoft SEAL 4.4.5；
- BLB 的 SCI 基线；
- BOLT 的非线性 SCI 实现。

依赖只写入仓库的 `.deps/`，不会安装到系统目录。

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

等价的一键入口是：

```bash
./scripts/reproduce.sh 4
```

完整测试包含真实的两方 conversion、MPC 非线性、Block 2、Softmax、Attention×V/LN2、FFN/GeLU、Block 5 和整体运行时。它明显慢于 framework-only 测试。

## 12 层 BERT-base 可变内部宽度配置

[`configs/bert_base_variable_dims_seq128.json`](configs/bert_base_variable_dims_seq128.json) 提供以下结构：

```text
tokens=128, model_width=768, q_heads=12, kv_heads=12
per-layer [qk_head_width, value_head_width, ffn_width]
[48,64,3072], [48,32,3056], [48,64,3072], [64,64,3056]
[64,64,3072], [64,64,2768], [64,48,3072], [48,64,2464]
[48,64,2144], [48,64,1232], [48,48,1232], [32,48,624]
```

这里的 QK/VO 宽度均为单个 head 的宽度。Wq/Wk/Wv/Wo/Wu/Wd 的非方阵尺寸由配置自动推导。

注意：当前真实 HERA runtime 只支持 Pre-LN，因此该文件是“BERT-base 外部形状 + 可变内部宽度”的 Pre-LN 网络，不是原始 BERT 的 Post-LN 拓扑。配置使用 `security_level=none` 的研究参数。

在服务器上只构建完整运行程序、不执行测试或推理：

```bash
cd Hera
./scripts/bootstrap_dependencies.sh .deps 8
source .deps/env.sh

cmake -S . -B build/bert-base-variable \
  -DCMAKE_BUILD_TYPE=Release \
  -DHERA_ENABLE_FULL_RUNTIME=ON \
  -DSEAL_DIR="$SEAL_DIR" \
  -DSCI_DIR="$SCI_DIR" \
  -DEigen3_DIR="$Eigen3_DIR"

cmake --build build/bert-base-variable \
  --target hera_private_inference --parallel 8
```

两台机器应使用相同 Git commit、配置、线程数和空闲端口区间。先在服务端启动：

```bash
./build/bert-base-variable/hera_private_inference \
  --config=configs/bert_base_variable_dims_seq128.json \
  --party=server --port=23100 --threads=8 \
  --weight-seed=2 --warmup=0 --repeat=1
```

然后在客户端启动：

```bash
./build/bert-base-variable/hera_private_inference \
  --config=configs/bert_base_variable_dims_seq128.json \
  --party=client --ip=SERVER_IP --port=23100 --threads=8 \
  --input-seed=1 --warmup=0 --repeat=1
```

上述 CLI 仍使用确定性合成权重和合成 `[128,768]` hidden states，用于结构与性能实验。真实 BERT checkpoint、token embedding、真实 padding mask 和分类头尚未接入模型文件加载器。12 层、`N=32768` 的离线 plaintext 准备与在线推理会占用大量内存和时间，正式运行前应先确认服务器资源和完整 `base` 到 `base+800` 端口区间可用。

## 两机运行

两台机器必须使用相同提交、配置和依赖锁。服务端先启动：

```bash
./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json \
  --party=server --port=23100
```

客户端随后连接：

```bash
./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json \
  --party=client --ip=SERVER_IP --port=23100
```

端口基址需保证到 `base+800` 的区间可用，并保持在 32768 以下。

## 仓库结构

```text
app/          CLI 入口
backend/      recording/unavailable backend
config/       JSON 解析与严格校验
configs/      tiny、conversion 和 Figure-10 风格配置
conversion/   复数 CKKS↔MPC 协议
core/         model/tensor 基础类型
graph/        类型化 Transformer DAG
he/           五个真实 HE/MPC stage executor
include/hera/ 公共接口
modules/      图构建模块
runtime/      整体编排、协调、abort/heartbeat
tests/        单元、协议和端到端测试
scripts/      依赖准备和复现脚本
```
