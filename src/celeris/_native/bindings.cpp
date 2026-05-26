// celeris production binding (pybind11): typed, NumPy-buffer-aware wrappers
// over the C ABI in celeris_core.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <stdexcept>
#include <string>

#include "celeris_core.hpp"

namespace py = pybind11;

// y[i] = a*x[i] + y[i], in place. x may be cast/copied; y is validated to be a
// real float64, C-contiguous, writeable buffer and is written IN PLACE. Taking
// y as a plain py::array (not py::array_t<double, c_style>) is deliberate: the
// templated form silently COPIES on wrong-dtype/non-contiguous input, losing
// the in-place write with no error. We validate y's buffer_info explicitly so
// any mismatch raises instead of silently corrupting results.
static void saxpy(double a,
                  py::array_t<double, py::array::c_style | py::array::forcecast> x,
                  py::array y) {
    auto xb = x.request();
    auto yb = y.request();                 // request() on a generic py::array
    if (xb.ndim != 1 || yb.ndim != 1)
        throw std::runtime_error("saxpy expects 1-D arrays");
    if (yb.format != py::format_descriptor<double>::format())
        throw std::runtime_error("y must be a float64 array");
    if (yb.readonly)
        throw std::runtime_error("y must be writeable");
    if (yb.strides[0] != static_cast<py::ssize_t>(sizeof(double)))
        throw std::runtime_error("y must be C-contiguous");
    if (xb.shape[0] != yb.shape[0])
        throw std::runtime_error("x and y must have equal length");
    celeris_saxpy(a, static_cast<const double*>(xb.ptr),
                  static_cast<double*>(yb.ptr), yb.shape[0]);
}

// Verify IR JSON and report the selected strategy (0=unsupported,1=hand-written,2=llvm).
static int compile_strategy(const std::string& ir_json) {
    char err[256];
    CelerisKernel* k = celeris_compile(ir_json.c_str(), err, sizeof(err));
    if (!k)
        throw std::runtime_error(err);
    int s = celeris_strategy(k);
    celeris_free(k);
    return s;
}

PYBIND11_MODULE(celeris_native, m) {
    m.doc() = "celeris native production binding (pybind11)";
    m.def("saxpy", &saxpy,
          "y[i] = a*x[i] + y[i] via the hand-written C++ kernel. y is validated "
          "(float64, C-contiguous, writeable) and written IN PLACE; mismatches raise.",
          py::arg("a"), py::arg("x"), py::arg("y"));
    m.def("compile_strategy", &compile_strategy,
          "Verify IR JSON; return selected strategy (0/1/2).",
          py::arg("ir_json"));
}
