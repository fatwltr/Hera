# Reproducibility contract

## What is pinned

`dependencies.lock` fixes the exact upstream revisions of SEAL, BLB and BOLT. The build rejects any SEAL version other than 4.4.5 and any HERA-SCI package without `HERA_SCI_ABI=3` and `OT_BACKEND=VOLE_SILENT_OT`.

The dependency pipeline is:

1. Build a static Release installation of SEAL 4.4.5 under `.deps/build/`.
2. Copy the pinned BLB SCI tree to a build directory.
3. Overlay the pinned BOLT `SCI/src` implementation.
4. Restore BLB's Ferret/PCG Silent-OT implementation and apply `third_party/hera-sci.patch` for the VOLE backend selection, odd-prime wrapping, SEAL 4.4.5 ABI alignment and cancellable networking.
5. Build and install the resulting HERA-SCI package under `.deps/build/`.
6. Configure HERA against only those two package locations.

Input repositories are never patched in place. The generated manifests record the selected ABI and SEAL revision.

## Test tiers

- `framework`: configuration, graph construction, contracts and recording execution. No cryptographic dependencies.
- `full`: all compiled cryptographic stages and protocol tests. Tests that use fixed ports run serially.
- `runtime_tiny`: a functional end-to-end profile intended for CI and local validation.
- `figure10_smoke`: a research profile whose degree and modulus chains require substantially more memory and time.

## Expected trust and security boundary

The code implements a semi-honest two-party research protocol. The client owns secret keys and private input; the server owns model parameters. The runtime does not decrypt intermediates. Test-only final reconstruction and deterministic model/input generation are not production interfaces.

Several supplied configurations deliberately use `security_level=none` to reproduce research schedules. Do not treat them as production parameters.

## Known numerical limitation

The CKKS→MPC one-way conversion regression target is two scale-13 ULP. The complete cross-profile CKKS→MPC→CKKS fixed-point FFT round trip currently permits up to 48 scale-13 ULP; reaching four ULP remains numerical-engineering work.

## Reproduction record

For an experiment, retain:

- the HERA Git commit;
- `dependencies.lock`;
- `.deps/build/seal-4.4.5/install/share/hera-seal/manifest.txt`;
- `.deps/build/hera-sci-abi-3/install/share/hera-sci/manifest.txt`;
- the JSON configuration;
- compiler/CMake versions, CPU model, thread count, party topology and command line.

The performance report separates setup, online inference and final output transfer. Synthetic weights are suitable for kernel/runtime benchmarking but are not a model-accuracy experiment.
