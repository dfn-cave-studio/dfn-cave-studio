"""Scientific tests for M9 fracture-size models."""

from types import SimpleNamespace

import numpy as np

from dfn_cave_studio.dfn.size_models import MLESizeModelFitter, sample_sizes, size_moments


def test_manual_distribution_moments_match_seeded_sampling():
    cases = [
        ("uniform", {}, 1.0, 4.0),
        ("truncated_lognormal", {"mu": 0.5, "sigma": 0.4}, 0.5, 4.0),
        ("truncated_power_law", {"exponent": 2.5}, 0.5, 4.0),
        ("truncated_exponential", {"rate": 0.8}, 0.5, 4.0),
    ]
    for name, parameters, lower, upper in cases:
        mean, mean2 = size_moments(name, parameters, lower, upper)
        values = sample_sizes(name, parameters, lower, upper, count=100_000, random_seed=42)
        assert np.isclose(values.mean(), mean, rtol=0.015)
        assert np.isclose(np.mean(values**2), mean2, rtol=0.025)


def test_mle_recovers_synthetic_lognormal_and_records_scores():
    samples = sample_sizes(
        "truncated_lognormal", {"mu": 1.0, "sigma": 0.3}, 1.0, 5.0, count=5000, random_seed=13
    )
    model = MLESizeModelFitter().fit(samples, domain_id=1, set_id=2, source_field="radius")
    lognormal = next(item for item in model.candidates if item.distribution_type == "truncated_lognormal")
    assert abs(lognormal.parameters["mu"] - 1.0) < 0.08
    assert abs(lognormal.parameters["sigma"] - 0.3) < 0.08
    assert model.source.value == "experimental"
    assert model.fit_status == "experimental"
    assert model.sample_count == 5000
    assert all(item.sample_count == 5000 for item in model.candidates)
    assert all(
        item.optimizer_message and (
            not item.converged or np.isfinite([item.log_likelihood, item.aic, item.bic]).all()
        )
        for item in model.candidates
    )


def test_failed_optimizer_is_explicit_and_does_not_publish_parameters(monkeypatch):
    monkeypatch.setattr(
        "dfn_cave_studio.dfn.size_models.optimize.minimize",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            message="forced optimizer failure",
            fun=float("nan"),
            x=np.array([0.0]),
        ),
    )
    model = MLESizeModelFitter().fit([1.0, 1.5, 2.0, 3.0], domain_id=1, set_id=1, source_field="radius")
    failed = [item for item in model.candidates if item.optimizer_message.startswith("optimization failed")]
    assert len(failed) == 2
    assert all(not item.converged for item in failed)
    assert all(item.parameters == {} for item in failed)
    assert all(item.log_likelihood is None and item.aic is None and item.bic is None for item in failed)
