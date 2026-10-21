# OpenPRISM native kernels

Optional Rust acceleration for the deterministic OpenPRISM renderer and thermal
local mean. Install from the OpenPRISM repository with a Rust toolchain:

```sh
python -m pip install ./native
```

Set `OPENPRISM_BACKEND=rust` before running OpenPRISM. The default remains
`numpy`. Installation builds a release-mode stable-ABI extension for Python
3.10 or newer; it does not require a Rust compiler at runtime.

See [installation, numerical contract and benchmarks](https://github.com/Alvaroinfantee/openprism/blob/main/openprism/docs/NATIVE_ACCELERATION.md).
