// celeris native runtime core (skeleton): C ABI, defensive IR verification,
// a pattern-matched golden kernel, and the documented LLVM lowering seam.
#include "celeris_core.hpp"

#include <cstring>
#include <string>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum Strategy { STRAT_UNSUPPORTED = 0, STRAT_HANDWRITTEN = 1, STRAT_LLVM = 2 };

struct CelerisKernel {
    int strategy = STRAT_UNSUPPORTED;
    std::string name;
};

// The native core must NOT trust its input: a corrupt/hand-crafted IR is
// rejected, never executed.
static bool verify_ir(const json& ir, std::string& why) {
    if (!ir.contains("name") || !ir.contains("params") || !ir.contains("body")) {
        why = "IR missing required top-level fields";
        return false;
    }
    if (!ir["body"].is_array()) {
        why = "body must be an array";
        return false;
    }
    return true;
}

// Toy "instruction selection": does this IR match the saxpy elementwise shape?
// A real backend lowers ANY verified IR; this skeleton recognizes one shape and
// routes it to the hand-tuned kernel. The Python-side kernel registry
// (backends/kernels.py) generalizes this idea with a fingerprint registry.
static bool matches_saxpy(const json& ir) {
    try {
        const auto& body = ir.at("body");
        if (body.size() != 1 || body[0].at("op") != "for") return false;
        const auto& loop = body[0];
        if (loop.at("body").size() != 1) return false;
        const auto& st = loop["body"][0];
        if (st.at("op") != "assign") return false;
        return st.at("target").at("k") == "index"
            && st.at("value").at("k") == "binop"
            && st["value"].at("op") == "+";
    } catch (...) {
        return false;
    }
}

// LLVM lowering seam (v0.4). Intentionally a stub — this is WHERE C++-side LLVM
// IR generation + ORC JIT goes, not an implementation. Built only with
// -DCELERIS_LLVM.
[[maybe_unused]] static bool lower_to_llvm(const json& /*ir*/, CelerisKernel& /*k*/,
                                           std::string& why) {
    why = "LLVM backend not built (compile with -DCELERIS_LLVM)";
    return false;
}

extern "C" {

CelerisKernel* celeris_compile(const char* ir_json, char* err_buf, int err_cap) {
    auto fail = [&](const std::string& m) -> CelerisKernel* {
        if (err_buf && err_cap > 0) {
            std::strncpy(err_buf, m.c_str(), static_cast<size_t>(err_cap - 1));
            err_buf[err_cap - 1] = '\0';
        }
        return nullptr;
    };

    json ir;
    try {
        ir = json::parse(ir_json);
    } catch (const std::exception& e) {
        return fail(std::string("bad JSON: ") + e.what());
    }

    std::string why;
    if (!verify_ir(ir, why)) return fail("verify failed: " + why);

    auto* k = new CelerisKernel();
    k->name = ir.value("name", "<anon>");

    if (matches_saxpy(ir)) {
        k->strategy = STRAT_HANDWRITTEN;
        return k;
    }
#ifdef CELERIS_LLVM
    if (lower_to_llvm(ir, *k, why)) {
        k->strategy = STRAT_LLVM;
        return k;
    }
#endif
    k->strategy = STRAT_UNSUPPORTED;
    return k;
}

void celeris_free(CelerisKernel* k) { delete k; }

int celeris_strategy(const CelerisKernel* k) { return k ? k->strategy : 0; }

// The one real, compiled kernel. -O3 autovectorizes this on most targets.
void celeris_saxpy(double a, const double* x, double* y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

}  // extern "C"
