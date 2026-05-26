// celeris native runtime core — C ABI declarations.
#pragma once
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to a compiled/selected kernel.
typedef struct CelerisKernel CelerisKernel;

// Compile IR (JSON) into a kernel handle. Returns NULL on failure (err_buf set).
CelerisKernel* celeris_compile(const char* ir_json, char* err_buf, int err_cap);

// Release a kernel handle.
void celeris_free(CelerisKernel* k);

// Selected strategy: 0=unsupported, 1=hand-written, 2=llvm-jit.
int celeris_strategy(const CelerisKernel* k);

// Hand-written golden kernel: y[i] = a*x[i] + y[i], i in [0, n).
void celeris_saxpy(double a, const double* x, double* y, int64_t n);

#ifdef __cplusplus
}  // extern "C"
#endif
