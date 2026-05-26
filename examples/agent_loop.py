"""agent_loop example — a tight numeric scoring kernel with a branch.

This stands in for the inner kernel of a lightweight agent loop: score a set of
candidates by a weighted sum of features, then apply a relu-style clamp so
negative scores are floored at zero. It is a pure numeric loop with a branch and
no external/LLM dependency, showcasing the kind of hot inner kernel celeris
targets.

celeris compiles the kernel on the first call and transparently falls back to
the original Python function if the kernel is unsupported on this machine; the
numeric result is identical either way, which is what we verify below.

Run it directly:  python examples/agent_loop.py
"""
import numpy as np

from celeris import fast_runtime
from celeris.types import F64Array


@fast_runtime
def score_candidates(
    weights: F64Array,
    features: F64Array,
    scores: F64Array,
    n_candidates: int,
    n_features: int,
) -> None:
    for c in range(n_candidates):
        s = 0.0
        for f in range(n_features):
            s = s + weights[f] * features[c * n_features + f]
        if s < 0.0:
            s = 0.0  # relu-style clamp: drop candidates that score negative
        scores[c] = s


def _reference(weights, features, n_candidates, n_features):
    feat = features.reshape(n_candidates, n_features)
    raw = feat @ weights
    return np.maximum(raw, 0.0)


def main():
    rng = np.random.default_rng(0)
    n_candidates = 256
    n_features = 8
    weights = rng.standard_normal(n_features)
    features = rng.standard_normal(n_candidates * n_features)
    scores = np.zeros(n_candidates, dtype=np.float64)

    score_candidates(weights, features, scores, n_candidates, n_features)

    expected = _reference(weights, features, n_candidates, n_features)
    np.testing.assert_allclose(scores, expected, rtol=1e-9, atol=1e-9)
    assert np.all(scores >= 0.0)
    print("agent_loop OK")


if __name__ == "__main__":
    main()
