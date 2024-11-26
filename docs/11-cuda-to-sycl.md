---
title:    Converting CUDA code to SYCL
subtitle: High-Level GPU Programming
date:     2024-11
lang:     en
---

# SYCLomatic / Intel DPC++ Compatibility Tool

- Automatic conversion of CUDA code to SYCL
- Included in oneAPI
- [Developer guide](https://www.intel.com/content/www/us/en/docs/dpcpp-compatibility-tool/developer-guide-reference/2025-0/overview.html)

# Examples

- Demo: daxpy

```bash
cd demos
dpct daxpy_cuda_hip.cpp
diff -u daxpy_cuda_hip.cpp dpct_output/daxpy_cuda_hip.cpp.dp.cpp
cd dpct_output
icpx -fuse-ld=lld -O3 -fsycl daxpy_cuda_hip.cpp.dp.cpp
```

- Demo: heat equation solver from CUDA to SYCL

# Hints

- Check the warnings emitted by the conversion tool!
- Use `dpct --help` to explore conversion options
- The generated code is specific to oneAPI as it uses functions from oneAPI `dpct` namespace
  - Alternative: `dpct --use-syclcompat` that uses [SYCLcompat header-only library](https://intel.github.io/llvm/syclcompat/README.html) instead

