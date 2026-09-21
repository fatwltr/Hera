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

## 环境

推荐 Ubuntu 22.04/24.04、x86-64、CMake 3.21+ 和支持 C++17 的编译器。

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git patch rsync \
  libssl-dev libgmp-dev libmpfr-dev libboost-dev
```

获取源码：

```bash
git clone https://github.com/fatwltr/Hera.git
cd Hera
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

仅测量密码运算开销、且机器无法容纳全部编码权重时，可以启用 benchmark-only
虚拟权重池：

```bash
./build/full/hera_private_inference \
  --config=configs/bert_base_variable_dims_seq128.json \
  --party=local --port=23100 --threads=8 \
  --weight-mode=virtual --weight-template-count=1
```

该模式仍执行原定数量的 rotation、`multiply_plain`、累加、rescale、
conversion 和 MPC 操作，但兼容的 Wq/Wk/Wv/Wo/Wu/Wd diagonals 会重复使用
有限数量的物理 plaintext。它用于低内存性能测试，不保持模型数值语义，也不能
作为模型精度结果。使用更大的 `--weight-template-count` 可以部分恢复真实的缓存/
内存访问压力。正常模式为 `--weight-mode=materialized`。

等价的一键入口是：

```bash
./scripts/reproduce.sh 4
```

完整测试包含真实的两方 conversion、MPC 非线性、Block 2、Softmax、Attention×V/LN2、FFN/GeLU、Block 5 和整体运行时。它明显慢于 framework-only 测试。

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
