# HERA private Transformer framework

This repository is a standalone, contract-first framework for composing a
private Transformer from HE and MPC kernels. It does not compile or link any
of the legacy CCMMTR project sources.

## Current boundary

The public inference boundary is ciphertext-to-ciphertext:

```text
HE hidden states -> HE/MPC/HE Transformer graph -> HE hidden states
```

The framework implements configuration, shape inference, a typed graph,
Figure-10-inspired placement, backend dispatch and contract checking. The
optional `hera_conversion` library implements the complex CKKS/MPC boundary.
The optional `hera_he` library implements Figure-10 Block 2 from refreshed
LayerNorm values through Q/K/V and encrypted QK scores. Other HE regions and
MPC nonlinear kernels are supplied by the staged optional libraries below.
`hera_private_runtime` now dispatches those implementations as one graph-wide
multi-layer Pre-LN runtime. The optional `hera_softmax` library implements the
next staged boundary from
Block-2 encrypted attention scores to MPC Softmax probability shares.
`hera_attention_ln2` continues from those shares through encrypted A*V, Wo,
the attention residual and LayerNorm rsqrt. `hera_ffn_gelu` then implements
the LayerNorm tail, Wu and the BLB-style split GeLU, ending in MPC activation
shares.
`hera_block5_bridge` converts those shares directly into the retained
attention-residual context, evaluates rectangular Wd, adds the HE residual
without a conversion, accurately realifies the result, and optionally runs
the next Pre-LN layer's Region-1 statistics and existing Block-2 executor.
The graph-wide runtime uses an explicit `SealSciPreLnEntryExecutor` instead of
that compatibility shortcut so every layer follows the same five-stage loop.

`UnavailableBackend` reports the first missing kernel with its graph node and
tensor contract. `RecordingBackend` creates opaque handles without computing
values; it exists only to test graph composition and dispatch order.

## Build and inspect

```bash
cmake -S . -B build
cmake --build build --target hera_private_inference hera_framework_tests
ctest --test-dir build -R hera_framework_tests --output-on-failure

./build/hera_private_inference --dry-run --dump-graph
./build/hera_private_inference --record-execution
```

Use `--config PATH` to select another JSON configuration. Unknown JSON fields
and invalid dimensions or HE profiles are rejected.

## Module contracts

- `LayerNormGraphModule` builds HE mean/variance operations, an MPC `rsqrt`,
  and explicit refreshes for values live across that boundary.
- `AttentionGraphModule` builds arbitrary-size Wq/Wk/Wv, grouped-head QK,
  A*V, Wo and the attention residual.
- `SoftmaxGraphModule` puts row max, clipping, sum, reciprocal and final
  normalization in MPC. Only `(1+x/64)^64` is evaluated in Block-3 HE.
- `ActivationGraphModule` describes MPC comparison/mux guards around an HE
  polynomial approximation.
- `FeedForwardGraphModule` builds Wu, activation, Wd and the FFN residual.

Every HE-region change is represented by explicit HE-to-MPC and MPC-to-HE
nodes. This is verbose by design: the concrete runtime can audit which live
values remain encrypted and which cross a refresh bundle.

## Remaining implementation work

The complete staged path is executable. Remaining engineering work, without
changing the graph modules, is:

1. Add a model-file loader and weight serialization format. The current API
   accepts in-memory weights and the CLI deliberately creates test weights.
2. Tighten the wide fixed-point FFT round-trip error from the current
   regression bound to the four-ULP target.

The real runtime now requires `PrivateTransformer::Prepare()` before
`Evaluate()`. Prepare compiles and freezes all model-dependent LayerNorm,
Q/K/V/Wo/Wu/Wd and bias plaintexts. A frozen-cache miss fails instead of
encoding during online inference. The dedicated `base+1` coordinator sends
two-second heartbeats, synchronizes every five-stage boundary, propagates
abort status, and cancels conversion plus all SCI sessions after peer failure.

Concrete backends must return tensors matching the graph-declared shape,
domain, layout, region, level and scale. The executor never calls a decrypt
operation, and no backend interface provides one.

## Whole Transformer runtime

Enabling the final bridge also builds `hera_private_runtime`, binds it through
`ITransformerExecutionRuntime`, and makes `PrivateTransformer::Evaluate()`
execute the concrete five-stage loop:

```text
Pre-LN entry / Block 2
 -> Softmax
 -> Attention x V / Wo / residual / LN2 rsqrt
 -> LN2 tail / Wu / split GeLU
 -> Wd / FFN residual / realify
 -> next layer or final HE hidden state
```

All stages share one `RuntimeTensorArena` and monotonic tensor-ID allocator.
The client owns all secret keys; the server stores ciphertext payloads and
model parameters. Between layers, the preceding shared-profile level-7 hidden
state is converted once to shares and independently refreshed to Region-1 L0
for statistics and shared Region-2/4 L0 for the new attention residual. It is
never incorrectly added directly to a next-layer level-6 value.

The ordered Softmax, LayerNorm-rsqrt and GeLU MPC calls share one
`SharedMpcSession`. Its sockets and Ferret/PCG Silent-OT state are prepared
once during setup and then reused across all layers. Conversion deliberately
keeps separate worker sessions because its workers execute concurrently and
must not interleave their stateful OT streams with nonlinear calls. Finished
stage inputs are erased from `RuntimeTensorArena` as soon as the next stage
has consumed them; only the caller-owned input and final output survive a
one-layer evaluation. This prevents ciphertext/share payload growth across
layers and repeated benchmark runs.

Build and run the small functional profile with:

```bash
cmake -S . -B build-runtime \
  -DHERA_ENABLE_BLOCK5_BRIDGE=ON \
  -DSEAL_DIR=/path/to/SEAL/lib/cmake/SEAL-4.4 \
  -DSCI_DIR=/path/to/hera-sci/install/lib/cmake/SCI
cmake --build build-runtime --target \
  hera_private_inference hera_private_runtime_tests -j

./build-runtime/hera_private_inference \
  --config=configs/runtime_tiny.json \
  --party=local --verify --port=23100
```

`--party=client|server --ip=HOST --port=PORT` runs the same protocol on two
machines. `--verify` is intentionally accepted only in local test mode and
decrypts only the final returned ciphertext. The control channel uses `port`,
the coordinator uses `port+1`, and conversion uses `port+10` with workers
from `port+100`. Softmax, LayerNorm and GeLU sequentially share the SCI
`IOPack` range beginning at `port+300`, including its internal `+50` and
`+100` sockets. Historical `port+500` and `port+700` ranges remain reserved
for deployment compatibility. The complete `base..base+800` range must stay
below 32768 so loopback clients cannot self-connect through Linux's ephemeral
port range.
The CLI reports
setup, online compute, final-output transfer, communication bytes and rounds
separately. `runtime_tiny.json` is a functional CI profile; the Figure-10
smoke configuration remains the resource-intensive research profile.

## Block 5 output and cross-layer bridge

Enable the final residual path together with its dependencies:

```bash
cmake -S . -B build-block5 \
  -DHERA_ENABLE_BLOCK5_BRIDGE=ON \
  -DSEAL_DIR=/path/to/SEAL/lib/cmake/SEAL-4.4 \
  -DSCI_DIR=/path/to/hera-sci/install/lib/cmake/SCI
cmake --build build-block5 --target hera_block5_bridge_tests -j
ctest --test-dir build-block5 -R hera_block5_bridge_tests --output-on-failure
```

`SealSciBlock5BridgeExecutor` requires the retained complex FFN residual from
`SealSciAttentionLn2Executor`. On the last layer it keeps the conventional
`L5 -> Wd L6 -> realify L7` output. On a non-final layer it converts GeLU
shares directly to the two-level terminal suffix, evaluates Wd one level
before terminal, locally mod-switches the retained residual to that exact
`parms_id`, adds it in HE, and lets realification land on terminal. The
residual itself never crosses an HE/MPC boundary. Context, scale, shape and
physical packing still have to match. The graph-wide runtime then performs
the next Pre-LN refresh; Post-LN automatic bridging remains unavailable.

## Block 2 HE backend

Enable the standalone SEAL backend without SCI:

```bash
cmake -S . -B build-block2 \
  \
  -DHERA_ENABLE_HE=ON \
  -DSEAL_DIR=/path/to/SEAL/lib/cmake/SEAL-4.4
cmake --build build-block2 -j
ctest --test-dir build-block2 -R hera_block2_he_tests --output-on-failure
```

`SealBlock2Executor` accepts two fresh Region-2 ciphertext tensors: centered
LayerNorm values and inverse standard deviation already broadcast to the same
physical spatial packing. It evaluates

```text
(centered * inverse_stddev) * gamma + beta -> Q,K,V -> QK^T
```

without a secret key or intermediate decryption. Independent real and
imaginary SIMD lanes are separated algebraically before the LayerNorm
elementwise product, preventing cross terms. Gamma is fused before one final
relinearize/rescale.

The caller registers ordinary row-major matrices. The backend compiles their
elementary column transformations directly into complex CKKS diagonals,
shares rotated inputs across Wq/Wk/Wv, folds `1/sqrt(qk_head_width)` into
Wq/bq, and emits native multi-head packing. All three matrices consume
`model_width`; their output widths are independent. GQA and MQA retain compact
logical K/V shapes and replicate values only in physical query-head lanes.

QK conjugates K so each complex product accumulates two reduction dimensions,
rotates K by attention diagonal, reduces packed lanes, applies tail masks, and
pairs successive diagonal groups into the real and imaginary parts of the
output ciphertext. Its level contract is suffix-based:

```text
fresh inputs 0 -> LayerNorm tail 1
V projection 2 (retained for Attention x V)
Q/K projection terminal-5 -> K preprocessing -> ct-ct product
-> alignment -> recovery -> output pairing at terminal
```

An eight-prime standalone Block-2 profile therefore retains the original
`Q/K L2 -> scores L7` path, while the ten-prime shared profile uses
`Q/K L4 -> scores L9` and avoids a later L7-to-L9 conversion-only drop. The Block-2
test covers a non-square GQA case (`Wq=4x4`, `Wk=4x2`, `Wv=4x3`), nonzero
biases, independent real/imag input lanes, client-generated evaluation keys,
server-side evaluation, final-only decryption, and level/accuracy checks.

## Complex CKKS/MPC conversion

Enable the optional protocol library with explicit SEAL and HERA-SCI package
locations:

```bash
cmake -S . -B build-conversion \
  \
  -DHERA_ENABLE_CONVERSION=ON \
  -DSEAL_DIR=/path/to/SEAL-4.4.5/install/lib/cmake/SEAL-4.4 \
  -DSCI_DIR=/path/to/HERA-SCI/install/lib/cmake/SCI \
  -DEigen3_DIR=/path/to/eigen/cmake
cmake --build build-conversion -j
ctest --test-dir build-conversion --output-on-failure
```

The protocol has four layers:

- `HeContextRegistry` creates one independent CKKS context and key set for
  every configured HE profile. Only the client registry contains secret keys.
  Serialized public/relinearization/Galois-key bundles support two-machine
  startup.
- `CkksMpcConverter` implements the low-level two-party conversion. It returns
  separate physical real and imaginary arithmetic-share tensors.
- `PhysicalSlotMap` records how ciphertext/slot/real-or-imaginary lanes map to
  logical tensor coordinates. Changing polynomial degree is performed by
  compacting/scattering local shares, never by converting ciphertexts across
  contexts.
- `SealSciConversionBackend` and `CryptoTensorStore` adapt the real/imag pair
  to the graph's single logical TensorHandle without exposing payloads in the
  graph API.

CKKS-to-MPC always moves a ciphertext copy to its single-prime terminal data
level. The server rejection-samples an unbiased polynomial mask `r_A mod q`,
adds it in NTT form, and sends only the masked ciphertext to the client. After
field-to-ring and complex fixed-point Decode, the server independently samples
`z_R` and `z_I` in `Z_(2^ell)` and refreshes real and imaginary output shares.
The three streams have distinct domain labels and a fresh session nonce.

MPC-to-CKKS performs complex fixed-point Encode locally and securely extends
43-bit shares into the statistical workspace. Following BLB's conversion
boundary, each party converts an encoded coefficient share once into the
composite target field `Q = product(q_j)`. SEAL then decomposes that full-Q
share locally into every RNS prime and applies the corresponding NTT. This
implementation deliberately keeps Q as a multi-limb integer; BLB's published
artifact exposes Q through a `uint64_t`, which is insufficient for a general
multi-prime target level. The client symmetrically encrypts its full-RNS
plaintext share and the server adds its own plaintext share. The protocol
never rewrites a ciphertext scale to disguise a numerical mismatch.

`coefficient_modulus_bits` denotes the ciphertext data-prime chain. The
registry appends a separate SEAL key-switch special prime, so SEAL does not
silently remove one configured Figure-10 data level from
`first_context_data()`.

The current security model is semi-honest two-party. The client is P0 and owns
the secret key; the server is P1 and owns HE ciphertexts. Production masking
uses `getrandom`-seeded HMAC-SHA256 streams. Deterministic randomness is
available only as a test dependency and is not a CLI option.

Run the conversion smoke test locally or on two machines:

```bash
./build-conversion/hera_complex_conversion \
  --party=local --config=configs/conversion_smoke.json

# machine serving ciphertext computation
./hera_complex_conversion --party=server --config=CONFIG \
  --profile=PROFILE --port=19100

# secret-key client
./hera_complex_conversion --party=client --config=CONFIG \
  --profile=PROFILE --ip=SERVER_IP --port=19100
```

The smoke test covers a complex ciphertext, independent A/Z-real/Z-imag
masks, a 43-bit/scale-13 sharing ring, full-RNS reconstruction, public-key
wire exchange and graph-backend payload adaptation. With the current
fixed-point local-truncation FFT, CKKS-to-MPC meets the two-ULP test target,
but the complete N=2048/scale-20 to N=4096/scale-24 cross-profile round trip
is bounded in regression tests at 48 scale-13 ULP rather than the desired
four ULP. Tightening this numerical bound
requires improving the wide-share FFT/truncation schedule; the implementation
does not hide it with plaintext reconstruction.

The inverse transform follows BLB's delayed-normalization schedule: the first
`log2(N)-1` butterfly levels grow in the wide ring and the public `1/N` factor
is folded into the final level. This avoids an additional one-bit local
truncation at every inverse-transform level. The MSE values in BLB Table 6 are
for the CKKS evaluation of the five HE blocks, not a CKKS/MPC round-trip ULP
bound; the artifact does not provide a four-ULP conversion regression test.

## Staged Softmax backend

Enable the concrete staged Softmax together with its conversion and HE
dependencies:

```bash
cmake -S . -B build-softmax \
  -DHERA_ENABLE_SOFTMAX=ON \
  -DSEAL_DIR=/path/to/SEAL/lib/cmake/SEAL-4.4 \
  -DSCI_DIR=/path/to/HERA-SCI/install/lib/cmake/SCI \
  -DEigen3_DIR=/path/to/eigen/cmake
cmake --build build-softmax --target hera_softmax_tests -j
ctest --test-dir build-softmax -R hera_softmax_tests --output-on-failure
```

`SealSciSoftmaxExecutor` has this concrete boundary:

```text
Block-2 CKKS attention diagonal
 -> CKKS-to-MPC -> MPC row max/mask
 -> MPC-to-CKKS(Block 3)
 -> HE /64, +1, six squarings
 -> CKKS-to-MPC
 -> MPC clip, row sum, reciprocal, smul
 -> 43-bit/scale-13 unsigned MPC probability shares
```

The output shape is `[tokens,q_heads,tokens]` with `kSoftmaxRows` layout.
Block-2 real and imaginary diagonal lanes are compacted locally after
conversion. Block 3 uses real slots only; a non-power-of-two token count gets
a next-power-of-two physical row stride, unused real slots are zero, and all
imaginary lanes are zero. The mapping supports row tails and multiple
ciphertexts.

Block 3 validates its configured modulus chain against the JSON exponent
schedule. The default `/64 + six squares` schedule requires seven consumed
levels and eight data primes, ending at the single-prime conversion level.
The kernel uses the actual dropped prime for every plaintext scale and never
repairs an invalid scale by arbitrary metadata rewriting.

The MPC path uses SCI compare/mux and B2A primitives for an exact restoring
fixed-point reciprocal, followed by secure multiplication and truncation.

## LayerNorm tail, Wu and split GeLU

Enable the Region-5 stage and all of its conversion dependencies with:

```bash
cmake -S . -B build-ffn \
  \
  -DHERA_ENABLE_FFN_GELU=ON \
  -DSEAL_DIR=/path/to/SEAL/lib/cmake/SEAL-4.4 \
  -DSCI_DIR=/path/to/HERA-SCI/install/lib/cmake/SCI \
  -DEigen3_DIR=/path/to/eigen/cmake
cmake --build build-ffn --target hera_ffn_gelu_tests -j
ctest --test-dir build-ffn -R hera_ffn_gelu_tests --output-on-failure
```

`SealSciFfnGeluExecutor` accepts signed 43-bit/scale-13 `centered` and
`inverse_stddev` arithmetic shares. It performs two independent MPC-to-CKKS
conversions, locally broadcasts inverse standard deviation over the
column-first physical packing, and evaluates:

```text
(centered * inverse_stddev) * ln2.gamma + ln2.beta
 -> Wu + bu -> realify -> HE F0/F1
 -> one batched CKKS-to-MPC -> three cmp + three mux
 -> signed 43-bit/scale-13 GeLU shares
```

The MPC implementation uses HERA-SCI's VOLE/Silent-OT backend, implemented by
the Ferret/PCG correlated-OT extension. The HE
polynomial shares `x^2`, `x^3` and `x^4` between both branches:

```text
F0 = a*x^4 - b*x^3 + c*x^2 + (0.5-d)*x + e
F1 = a*x^4 + b*x^3 + c*x^2 + (0.5+d)*x + e
```

Only interval comparisons and branch selection run in MPC. The output remains
an arithmetic share; conversion for Wd is deliberately the next stage.

The LayerNorm affine operation retains a truthful `2^60` intermediate scale.
Forcing it immediately to `2^45` after a single lazy rescale would leave only
about 15 bits for the gamma plaintext and does not meet the `1e-3` accuracy
contract. Wu consumes the `2^60` input with a `2^45` plaintext and returns to
`2^45` without an extra level. Realification multiplies the two conjugation
branches by high-precision `1/2` and `-i/2`, consumes one 46-bit level, and
returns accurate real-only ciphertexts at `2^46` before the GeLU powers.

SEAL removes data primes from the end of `coefficient_modulus_bits`. Region 5
therefore enters at `terminal-6` and uses the physical drop schedule
`60,46,46,46,46,60`: LayerNorm, Wu, realification, two shared-power stages,
and coefficient evaluation. With the default eight-prime chain this is
`input L1 -> LN L2 -> Wu L3 -> realify L4 -> F0/F1 L7 terminal`.
Only the shallow original-`x` branch is locally mod-switched to L7 for the MPC
comparison batch. The terminal level is derived from the chain and
`terminal_prime_count`, never hard-coded.

The regression test covers column-first real/imaginary packing, non-square
Wu, optional bias, HE branch accuracy, threshold boundary behavior, exact HE
levels, server secret-key absence, and a full two-process
MPC-to-CKKS-to-HE-to-MPC-to-GeLU execution. Decryption/reconstruction occurs
only in test code after the stage finishes.
The unified HERA-SCI package is required for conversion and nonlinear builds.
It retains BOLT's two-direction COT API but dispatches both directions through
BLB's VOLE/Silent-OT implementation; this API is required by BLB's
reciprocal-square-root implementation. Softmax multiplication now uses SCI's
native `MultMode::None` secret-share product over the independent straight and
reversed Ferret transports. The patched boolean wire format and Ferret delta
handling are covered by dedicated comparison, mux, batched multiplication and
1-out-of-N OT tests. These paths operate on shares and do not reconstruct an
intermediate value. CMake rejects a SCI installation without the ABI-3 VOLE
manifest marker.

The upstream Ferret construction still bootstraps its base correlated OTs once
with emp-ot's IKNP primitive; all HERA online OT extension calls use the
Ferret/PCG Silent-OT objects. Eliminating even this setup-only bootstrap would
require replacing Ferret's base-COT construction and is not part of this
backend switch.

The execution report counts the two CKKS-to-MPC calls, one internal
MPC-to-CKKS call, HE multiplications/rescales, MPC operations, conversion and
MPC communication/rounds, mask bytes, and internal mod-switches. There is no
V argument and no final MPC-to-CKKS conversion in this phase:

```text
SOFTMAX_TO_MPC_READY=YES
SOFTMAX_MPC_TO_HE_READY=NO
SOFTMAX_AV_INTEGRATION_READY=NO
```

## Attention value through LayerNorm reciprocal square root

Enable the third concrete stage with the repository-built HERA-SCI package:

```bash
scripts/prepare_seal.sh /path/to/SEAL-v4.4.5 build/seal-4.4.5 4
scripts/prepare_hera_sci.sh /path/to/BLB/SCI /path/to/BOLT/SCI \
  "$PWD/build/seal-4.4.5/install/lib/cmake/SEAL-4.4" build/hera-sci-abi-3 4
cmake -S . -B build-attention-ln2 \
  \
  -DHERA_ENABLE_ATTENTION_LN2=ON \
  -DSEAL_DIR=$PWD/build/seal-4.4.5/install/lib/cmake/SEAL-4.4 \
  -DSCI_DIR=$PWD/build/hera-sci-abi-3/install/lib/cmake/SCI \
  -DEigen3_DIR=$PWD/build/hera-sci-abi-3/SCI/extern/eigen/build
cmake --build build-attention-ln2 --target hera_attention_ln2_tests -j
```

`SealSciAttentionLn2Executor` implements this boundary without an internal
decrypt or plaintext bridge:

```text
43/13 MPC Softmax shares
 -> MPC-to-CKKS at the retained V ciphertext's exact level 2 parms_id
 -> complex-packed A*V (levels 2 -> 5)
 -> rectangular Wo + optional bo (level 6)
 -> mod-switch and add the same-context attention residual
 -> z+conj(z), (z-conj(z))*(-i) realification
 -> HE mean/center/square/variance+epsilon (levels 7 -> 9)
 -> two CKKS-to-MPC conversions
 -> BLB/BOLT SCI reciprocal square root
 -> centered and inverse-stddev 43/13 arithmetic shares
```

Region 2 and Region 4 must name the same HE profile. The supplied smoke
configuration uses ten data primes and depth nine; Softmax A is created at
the exact `parms_id` of retained V rather than matched by level number or bit
width. Residual addition accepts a higher-level ciphertext only in the same
context, mod-switches it to the Wo output, and rejects scale or packing
mismatches.

The A/V planner supports MHA, GQA and MQA physical head replication, head
tiles, value-width tails and non-power-of-two token counts. Wo emits complex
strict columns; realification produces real-only strict columns. LayerNorm
folds valid-feature masks into its existing mean and variance plaintext
multiplications, preventing unused tail-slot approximation noise from entering
the reductions without consuming another modulus level.

The rsqrt API remains 43-bit/scale-13. Each layer selects its internal ring
with `attention_norm_rsqrt_bits` (20 by default). Its required public bound is
`variance < 2^(attention_norm_rsqrt_bits - scale_bits - 1)`; therefore the
default is `variance < 64`. The implementation reduces both additive shares
to that ring, runs the BLB rsqrt circuit, and securely zero-extends the
positive result back to 43 bits. A model with a wider variance range can raise
the setting up to `numeric.mpc_ring_bits` without changing the stage API.
