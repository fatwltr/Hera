# Third-party dependencies

HERA does not vendor the following upstream repositories. The reproducibility script fetches pinned revisions into the ignored `.deps/` directory.

| Dependency | Pinned source | Purpose |
|---|---|---|
| Microsoft SEAL | `microsoft/SEAL` 4.4.5 | CKKS runtime |
| BLB | `best-orange/blb` | SCI baseline, conversion extensions and Ferret/PCG Silent-OT backend |
| BOLT | `cmu-cryptosystems/BOLT` | Validated SCI nonlinear implementation |

SEAL and BOLT publish their own license files. BLB did not expose a top-level license in the pinned repository revision when this dependency recipe was prepared. Because this repository applies and links code derived from those projects, verify all upstream license obligations before public redistribution.

`third_party/hera-sci.patch` is applied only to a generated dependency copy. No generated dependency source or binary is intended to be committed.

The SCI recipe uses its vendored EMP headers and the OS random source. Its
communication directions are explicit tasks, including with one OpenMP thread.
`scripts/seal_rotation_hoisting.patch` and `scripts/seal_system_zstd.patch` are
also applied only to a generated SEAL copy. Compression links the system
Zstandard or zlib development package; no machine-global SEAL, SCI or EMP
installation is required.
