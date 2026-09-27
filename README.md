# HERA

HERA 是研究用途的 CKKS/MPC 私有 Transformer 推理原型，支持多层 Pre-LN +
GeLU、MHA/GQA/MQA 和可变的 QK、V、FFN 宽度。输入和输出都是 CKKS 密文。
当前配置允许 `security_level=none`，**不能用于生产安全部署**。

## 在 Ubuntu 服务器上构建并验证

推荐 Ubuntu 22.04/24.04、x86-64。首次构建需要访问 GitHub 和 GitLab；
若 HERA 仓库是私有的，先为服务器配置 GitHub SSH 访问权限，或改用已认证的
HTTPS 克隆。以下命令不依赖预先设置的 SEAL/SCI 环境变量或开发机路径：

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

`bootstrap_dependencies.sh` 在仓库内下载并构建固定版本的 SEAL 4.4.5 和
HERA-SCI，自动生成 `.deps/env.sh`；`build_and_test.sh` 读取该文件、构建程序
并运行全部 CTest。无需手工设置 `SEAL_DIR`、`SCI_DIR` 或 `Eigen3_DIR`。
两个 `2` 是并行编译任务数，可按内存增加。成功时应看到
`HERA_BUILD_STATUS=PASS` 和 `HERA_PRIVATE_INFERENCE_STATUS=PASS`。
`--party=local` 在本机启动客户端和服务端；请确保端口 23100 起的协议端口空闲。

先用 `runtime_tiny.json` 验证安装。128-token 或 12 层配置需要显著更多内存和
时间，不适合作为首次运行测试。服务器只能获取已经提交并推送的代码；本机
未提交的改动不会随 `git clone` 出现。

若只需要不依赖 SEAL/SCI 的图框架检查：

```bash
./scripts/build_and_test.sh framework 2
./build/framework/hera_private_inference \
  --config=configs/figure10_smoke.json --dry-run --dump-graph
```

## 单算子 benchmark

完成上述 full 构建后，下列程序均已编译。若只想构建 benchmark、跳过完整
运行时的 CTest，在依赖准备完成后执行：

```bash
cmake --preset full
cmake --build build/full --target \
  hera_bench_qk hera_bench_attention_v \
  hera_bench_he_functions hera_bench_mpc_functions \
  hera_bench_conversion -j 2
ctest --test-dir build/full -R 'hera_bench_.*_smoke' --output-on-failure
```

`full` preset 使用仓库内的 `.deps/`，不包含固定的用户主目录路径。若依赖
放在别处，请使用 `./scripts/build_and_test.sh full 2 <依赖目录>`。

QK、Attention×V 从已按内核要求打包的密文开始计时：

```bash
./build/full/hera_bench_qk \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2
./build/full/hera_bench_attention_v \
  --tokens=32 --heads=8 --width=16 --degree=8192 \
  --warmup=1 --iterations=10 --he-threads=2
```

其余 HE 投影、多项式及真实双进程 MPC/转换测试：

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

`--warmup` 为不计时的预热次数，`--iterations` 为计时次数；`--degree` 为
CKKS 多项式次数，`--he-threads` 为 HE 线程数，转换程序的 `--threads` 为
协议 worker 数。所有 benchmark 都在计时区间外准备输入并检查最终结果。
具体边界和 BLB 可比性见 [QK/Attention×V 说明](docs/benchmarks/he_qk_attention_v.md)
及 [其他模块说明](docs/benchmarks/functional_kernels.md)。

## 配置与其他说明

- [单层 BERT-base 配置](configs/bert_base_single_layer_seq128.json) 和
  [12 层可变宽度配置](configs/bert_base_variable_dims_seq128.json) 仅用于
  资源允许的研究测试；先用 `--dry-run` 检查配置。
- 仅测时间且内存不足时，可使用 `--weight-mode=virtual` 和
  `--weight-template-count=1` 复用合成 plaintext。该模式不保持模型数值语义，
  不能用于精度评估。
- 模型 plaintext 在 `Prepare()` 阶段编码；正常 online 计时不含权重编译。
  `--he-threads`、`--conversion-workers` 和 `--he-memory-mib` 分别控制 HE
  并行、转换 worker 与 HE 临时任务内存预算。

协议、level/packing 与运行限制见 [实现说明](docs/IMPLEMENTATION.md)和
[复现说明](docs/REPRODUCIBILITY.md)；第三方来源及许可证注意事项见
[THIRD_PARTY.md](THIRD_PARTY.md)。
