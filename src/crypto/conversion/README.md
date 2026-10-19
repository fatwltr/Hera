# Hera 工程化 BLB/M&M 转换

`BlbReferenceConverter` 是 Hera 唯一的 CKKS↔MPC 转换执行器。旧 `CkksMpcConverter` 及其源文件已移除，没有兼容别名，也没有通用比较转换回退。公共接口在 `include/hera/blb_reference.h`；数据格式、配置和计时结构仍在 `include/hera/conversion.h`。

## 固定参考版本

| 参考 | 采用内容 | 固定版本 |
|---|---|---|
| [BLB 论文](https://www.usenix.org/system/files/usenixsecurity25-xu-tianshi.pdf)，§6.3、Appendix C.3 | Algorithms 1/2；先统计扩环、后本地 R2F 校正 | USENIX Security 2025 |
| [BLB 工件](https://zenodo.org/records/15590214) | 作者提供的工件入口；本次没有把它当作完整生产实现的来源 | record 15590214 |
| [SEC-PPDL FixPoint.h](https://github.com/Tianshi-Xu/SEC-PPDL/blob/00e8223ec5d3a024754209e762068355463a4f09/Operator/NonlinearOperator/include/NonlinearOperator/FixPoint.h) | `Ring2Field` / `Field2Ring` / `extend(msb_zero)` 的接口和环域转换结构 | `00e8223ec5d3a024754209e762068355463a4f09` |
| [SPU ckks_mpc_test.cc](https://github.com/fionser/spu/blob/78ba12588b51bdfa298a4bbf95f6c84ab801c00b/libspu/mpc/cheetah/rlwe/ckks_mpc_test.cc#L999) | 本地 Encode/Decode、转换顺序、宽环编码的限制 | `78ba12588b51bdfa298a4bbf95f6c84ab801c00b`，分支 `dev/ckks_mpc` |
| [OpenMM](https://github.com/CPS4AI/OpenMM/tree/9e8e73d15fb6e963dff7d8e244404fa127173d90)，SCI `BuildingBlocks/truncation.cpp`、`value-extension.cpp` | 已知重构值 MSB 时，用双方本地高位得到 wrap，替代全宽比较 | `9e8e73d15fb6e963dff7d8e244404fa127173d90` |

上述三个代码仓库的 HEAD 于 2026-10-18 用 `git ls-remote` 核实。源码快照、SHA256 和构建/验证记录保存在 `build/full/verification/blb_reference_rebuild_2026-10-18/`。快照是审查材料，不参与编译。

参考中有几个不能直接照搬的细节：SPU 的 `ModulusExtend` 同时拿到双方份额，属于模拟；该测试还明确说明本地 Encode 需要较大的环。其编码器的槽位排列与 SEAL 当前使用的排列也不同。这里沿用已验证的 SEAL 顺序 FFT，用真实 SCI OT 实现扩环。SEC-PPDL 的 `Ring2Field` 将总位宽固定到 60，本实现将统计扩展长度作为参数，默认 κ=40，不把固定 60 位视作所有参数下都提供 40 位统计裕量。

## 调用链与工程能力

```text
PrivateTransformer / HE 阶段
  → SealSciConversionBackend / 各融合阶段的转换入口
    → BlbReferenceConverter
      → RingProtocol（quarter-range 算术 OR OT）
      → FixedCkksTransform / EncodePreparedSlots
      → 本地 RNS 投影 + SEAL 加解密/密文传输
```

主实现只有 `src/crypto/conversion/blb_reference.cpp`。原生产执行器的批处理、持久 worker、SEAL 槽位映射、实虚部打包、广播份额去重、padding 跳过、目标 HE profile/level/scale、FFT/RNS 缓存、内存预算、异步本地任务和阶段计时均已整合。转换 worker 只初始化一个独立的前向 OT 池，不再初始化反向池、KKOT 包装或 `FixOp` 比较对象。`blb_ring_protocol.cpp` 不含全宽秘密比较或 compare→B2A 的转换代码。

`SealSciConversionBackend`、LayerNorm、Softmax、GeLU、跨层 bridge 和私有推理入口都使用这个类型。源码调用方应包含 `hera/blb_reference.h`，并把原类型名改为 `BlbReferenceConverter`；构造参数仍是 role、ConversionConfig、context registry、transport、randomness、peer address、worker port、worker count、compression。

## 模型推理直接模拟

模型 CLI 对物化权重和虚拟权重统一设置 `ConversionConfig::benchmark_unchecked_range=true`，直接执行转换，不要求范围声明或范围证明，也不自动声明范围成立：

```bash
./build/full/hera_private_inference \
  --config=configs/runtime_tiny.json --party=local --threads=2 --verify
```

两种权重模式均直接模拟执行真实 HE/MPC/OT、编码、扩环和密文传输；`--verify` 在物化权重下保留显式请求的最终输出有限性检查。

## 底层转换 API 的数值契约

唯一支持的协议要求：

| 阶段 | 对重构秘密整数的要求 |
|---|---|
| C2M F2R | 含 HE 误差的中心多项式系数 `|a_i| < floor(q/4)` |
| M2C 槽位准备 | MPC 定点整数 `|x_int| < 2^(l−2)` |
| M2C 系数扩环 | Encode/截断后的整数系数在自己的 k 位环中满足 `|c_i| < 2^(k−2)` |

个体随机份额无需满足小范围。协议不会重构秘密检查数值，也不会因为虚拟权重而自动声明满足。调用方可以：

1. 在 `ConversionConfig` 中显式设置 `public_c2m_quarter_range`、`public_m2c_quarter_range`，声明相应方向的所有入口符合上述前提；或
2. 给 C2M 提供足够紧的 `public_coefficient_bound_bits`，给 M2C target 提供 `public_value_bound_bits` / `public_ciphertext_bound_bits`，由公开 FFT 上界推导覆盖两段 M2C 扩环。

这些布尔声明默认 false。通常没有声明且不能由公开 metadata 证明时，双方在进入该调用的转换 OT 前拒绝；不会退回通用算法。虚拟权重性能模拟的显式策略见下一节。OT 会话初始化本身仍可能在构造 converter 时发生。无效/不匹配的 metadata、prepared 宽度及方向也在公开协议帧中协调拒绝。

底层转换 API 的默认契约检查用于独立正确性验证，不作为模型 CLI 模拟运行的范围前置条件。

## 模拟执行策略

模型 CLI 始终设置 `ConversionConfig::benchmark_unchecked_range=true`。该策略不把 C2M/M2C 的公开范围证明作为执行前提，允许模拟数据超出 quarter-range 而继续运行。它不把 `public_c2m_quarter_range` / `public_m2c_quarter_range` 改成 true，也不会把 C2M 的模拟输出标成有证明的范围。

仍使用同一个有界算法、相同环宽和 κ、实际 HE/MPC/OT。没有恢复通用转换，没有裁剪输入、替换结果或跳过协议。因此所得数值不作为转换正确性、模型精度或真实数据安全性的验证结果。双方模式也绑定在 worker 握手和每次调用帧中；结构、布局、位宽、profile/level、公开参数一致性检查继续执行。

模型 CLI 只输出 HE、MPC、conversion 的在线时间和通信量，以及双方标识和执行状态，不输出准备阶段时间和内部诊断字段。虚拟权重下跳过模型数值验证，包括 `--verify` 的最终输出有限性检查；`--verify` 的本地运行限制仍保留。

物化权重同样使用 `BENCHMARK_UNCHECKED`。底层 C++ 转换 API 的模拟策略仍默认关闭，独立调用方可显式设置该字段。

```bash
./build/full/hera_private_inference \
  --config=configs/bert_base_single_layer_seq128.json \
  --party=local --threads=2 --weight-mode=virtual --verify
```

## 协议流程

C2M：服务端切换到单末端素数 q，均匀采样系数掩码 A，发送 `ct+A` 并保留 `−A`；客户端解密、逆 NTT。双方执行有界 F2R，然后在宽环中本地 Decode/截断，得到实虚部 MPC 份额。只有客户端持有解密密钥。

M2C：原始 l 位份额先安全准备到 l+κ 位宽环，再本地 Encode，保留系数精度；系数进行新鲜统计扩环，然后本地对目标每个 RNS 素数投影，一方减去对应 `2^w mod q_j`；客户端加密自己的多项式份额，服务端同态加上另一份额。

二进制环中，一方加 `2^(l−2)` 后重构值的 MSB 已知为零，进位为双方本地 MSB 的 OR。素数环中加 `floor(q/4)`，进位为 `[u>=ceil(q/2)] OR [v>=ceil(q/2)]`。加权算术 OR 使用新鲜 OT，仍有处理、通信和 OT 补充成本。

κ 默认 40，未删统计扩环或缩小原环。固定 R2F 校正依赖新鲜扩环后的分布；不能把任意 FFT 输出直接套用固定校正。本地截断、FFT 和 R2F 有数值/统计假设，测试不是完整安全证明；κ 也不直接等于整模型联合失败概率。

## prepared 入口

`MpcToCkksPrepared(const PreparedBlbSlots&, const MpcToCkksTarget&)` 与普通 M2C 共用同一批处理/Encode/RNS/加密流水线。输入覆盖 target 的完整物理实虚两平面，位宽必须为 `l+κ`、scale 为配置的 MPC scale；重构值仍应满足原始 MPC 数值域契约。不能只修改窄环份额的位宽标签或补零。

该入口跳过槽位扩环，仅保留 R2F 内的系数扩环。现有 Hera 非线性算子输出仍是 l 位份额，因此正常模型路径继续使用完整 M2C。prepared 入口不会自动让上游准备免费；`slot_extension_input_elements`、`coefficient_extension_elements` 和阶段计时会分别反映实际执行。

prepared 的位宽、元素数量、未使用高位、公开零虚部与广播周期声明会在进入 OT 前检查。prepared/raw 的区别绑定在协议帧中，不能让两方混用。

## 协议版本与验证

协议升级为 **v19**。双方必须重建到匹配版本；v18 及更早格式不兼容。worker 数、batch、数值精度、公开契约、目标参数与调用方向需要一致。

```bash
cmake -S . -B build/full -DHERA_ENABLE_FULL_RUNTIME=ON
cmake --build build/full -j2
env OMP_NUM_THREADS=2 ctest --test-dir build/full --output-on-failure
```

`tests/blb_reference_tests.cpp` 另校验故意超出 quarter-range 的模拟输入能执行转换、不传播虚假范围证明，同时仍拒绝非法位宽/布局。物化及虚拟 CLI 回归均不提供范围声明，检查双方在线统计和精简输出格式。原测试在真实两进程 OT 下校验有界整数转换、正负边界、非整批尾部、40–127 位目标环、多个 HE profile、原始/ prepared M2C、往返误差、缺失/不足/不匹配范围拒绝，以及拒绝后的正常恢复。生产回归还覆盖多 worker、混合布局、实虚部打包、广播去重、padding、单掩码、目标 profile/level 和整层运行时。测试专用 oracle 会重构输出，生产协议不重构。

通用转换专属的 wide-B2A 函数及对应测试目标已移除；算术 OR 的宽环边界测试保留。SCI 非线性运算仍可使用其自己的比较协议，这与已删除的 conversion 通用回退无关。

## 性能复现

```bash
cmake --build build/full --target hera_blb_reference_benchmark -j2
env OMP_NUM_THREADS=1 taskset -c 0,2,4,6,8,10 \
  build/full/hera_blb_reference_benchmark > benchmark.log
```

该 harness 已迁移到工程化执行器，测单 worker、单密文的 C2M / M2C / prepared M2C。默认每组至少 30 次，并覆盖三个前向 OT 缓冲补充周期。`--quick` 只用于接口/输出冒烟检查，不是摊销性能结论。完整转换计时包含 OT、FFT、加解密、协商及密文传输；同步、输入生成、验证、HE 密钥生成与会话初始化另计。prepared 的输入在计时外通过真实安全扩环准备。

之前的串行参考结果存于 `build/full/verification/blb_reference_cost_2026-10-18/`，属于整合前实现。它们不应标作当前工程化版本的测量结果。本次迁移记录和验证日志位于 `build/full/verification/blb_engineered_only_2026-10-18/`。
