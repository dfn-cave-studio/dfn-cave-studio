"""Fracture-size distributions and MLE fitting for M9."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np
from scipy import optimize, stats

from dfn_cave_studio.models.m9 import SizeFitCandidate, SizeModel, SizeModelSource


def truncated_power_moment(lower: float, upper: float, exponent: float, order: int) -> float:
    """Return E[R**order] for density proportional to r**(-exponent-1)."""
    if lower <= 0 or upper <= lower or exponent <= 0:
        raise ValueError("invalid truncated power-law parameters")
    normalizer = exponent / (lower ** (-exponent) - upper ** (-exponent))
    delta = order - exponent
    if abs(delta) < 1e-12:
        return normalizer * math.log(upper / lower)
    return normalizer * (upper**delta - lower**delta) / delta


def truncated_lognormal_moment(lower: float, upper: float, mu: float, sigma: float, order: int) -> float:
    """Return an exact raw moment of a doubly truncated lognormal."""
    if lower <= 0 or upper <= lower or sigma <= 0:
        raise ValueError("invalid truncated lognormal parameters")
    z0 = (math.log(lower) - mu) / sigma
    z1 = (math.log(upper) - mu) / sigma
    denominator = stats.norm.cdf(z1) - stats.norm.cdf(z0)
    shifted0 = (math.log(lower) - mu - order * sigma * sigma) / sigma
    shifted1 = (math.log(upper) - mu - order * sigma * sigma) / sigma
    return math.exp(order * mu + 0.5 * order * order * sigma * sigma) * (
        stats.norm.cdf(shifted1) - stats.norm.cdf(shifted0)
    ) / denominator


def truncated_exponential_moment(lower: float, upper: float, rate: float, order: int) -> float:
    """Return a numerical raw moment of a truncated exponential."""
    if lower <= 0 or upper <= lower or rate <= 0:
        raise ValueError("invalid truncated exponential parameters")
    from scipy.integrate import quad

    denominator = math.exp(-rate * lower) - math.exp(-rate * upper)
    value, _ = quad(lambda radius: radius**order * rate * math.exp(-rate * radius), lower, upper)
    return value / denominator


def size_moments(distribution_type: str, parameters: dict[str, float], lower: float, upper: float) -> tuple[float, float]:
    """Compute E[R] and E[R^2] for every supported M9 distribution."""
    if distribution_type == "fixed":
        radius = parameters.get("radius", lower)
        return radius, radius * radius
    if distribution_type == "uniform":
        return (lower + upper) / 2.0, (lower * lower + lower * upper + upper * upper) / 3.0
    if distribution_type == "truncated_lognormal":
        mu, sigma = parameters["mu"], parameters["sigma"]
        return (
            truncated_lognormal_moment(lower, upper, mu, sigma, 1),
            truncated_lognormal_moment(lower, upper, mu, sigma, 2),
        )
    if distribution_type == "truncated_power_law":
        exponent = parameters["exponent"]
        return truncated_power_moment(lower, upper, exponent, 1), truncated_power_moment(lower, upper, exponent, 2)
    if distribution_type == "truncated_exponential":
        rate = parameters["rate"]
        return truncated_exponential_moment(lower, upper, rate, 1), truncated_exponential_moment(lower, upper, rate, 2)
    raise ValueError(f"unsupported size distribution: {distribution_type}")


def sample_sizes(
    distribution_type: str,
    parameters: dict[str, float],
    lower: float,
    upper: float,
    *,
    count: int,
    random_seed: int,
) -> np.ndarray:
    """Sample radii reproducibly from a supported distribution."""
    rng = np.random.default_rng(random_seed)
    if distribution_type == "fixed":
        return np.full(count, parameters.get("radius", lower), dtype=float)
    if distribution_type == "uniform":
        return rng.uniform(lower, upper, count)
    if distribution_type == "truncated_lognormal":
        mu, sigma = parameters["mu"], parameters["sigma"]
        a, b = (math.log(lower) - mu) / sigma, (math.log(upper) - mu) / sigma
        return np.exp(stats.truncnorm.rvs(a, b, loc=mu, scale=sigma, size=count, random_state=rng))
    if distribution_type == "truncated_power_law":
        exponent = parameters["exponent"]
        u = rng.random(count)
        return (lower ** (-exponent) - u * (lower ** (-exponent) - upper ** (-exponent))) ** (-1 / exponent)
    if distribution_type == "truncated_exponential":
        rate = parameters["rate"]
        u = rng.random(count)
        return -np.log(math.exp(-rate * lower) - u * (math.exp(-rate * lower) - math.exp(-rate * upper))) / rate
    raise ValueError(f"unsupported size distribution: {distribution_type}")


class SizeModelFitter(ABC):
    """Replaceable interface for fracture-size inference."""

    @abstractmethod
    def fit(self, samples: Iterable[float], *, domain_id: int | None, set_id: int, source_field: str) -> SizeModel:
        """Fit and select a size model."""


class MLESizeModelFitter(SizeModelFitter):
    """MLE comparison of all continuous M9 size distributions."""

    def fit(self, samples: Iterable[float], *, domain_id: int | None, set_id: int, source_field: str) -> SizeModel:
        values = np.asarray(list(samples), dtype=float)
        values = values[np.isfinite(values) & (values > 0)]
        if values.size < 2:
            raise ValueError("at least two positive size observations are required")
        lower, upper = float(values.min()), float(values.max())
        if math.isclose(lower, upper):
            upper = float(np.nextafter(lower, math.inf))
        candidates = [self._uniform(values, lower, upper), self._lognormal(values, lower, upper)]
        candidates.extend([self._power(values, lower, upper), self._exponential(values, lower, upper)])
        selectable = [item for item in candidates if item.converged and item.aic is not None]
        if not selectable:
            messages = "; ".join(item.optimizer_message for item in candidates)
            raise RuntimeError(f"all experimental size fits failed: {messages}")
        selected = min(selectable, key=lambda item: item.aic)
        mean, mean2 = size_moments(selected.distribution_type, selected.parameters, lower, upper)
        proxy = source_field in {"trace_length", "mapped_length"}
        return SizeModel(
            domain_id=domain_id,
            set_id=set_id,
            distribution_type=selected.distribution_type,
            parameters=selected.parameters,
            min_radius=lower,
            max_radius=upper,
            mean_radius=mean,
            mean_squared_radius=mean2,
            source=SizeModelSource.EXPERIMENTAL,
            measurement_field=source_field,
            sample_count=int(values.size),
            candidates=sorted(candidates, key=lambda item: item.aic if item.aic is not None else math.inf),
            fit_status="experimental",
            converged=selected.converged,
            optimizer_message=selected.optimizer_message,
            provenance={
                "calibration_only": True,
                "measurement_is_radius": not proxy,
                "size_proxy": proxy,
                "reliability": "EXPERIMENTAL",
                "limitations": [
                    "truncation bounds use sample extrema",
                    "truncated lognormal parameters are not a full truncated-likelihood optimum",
                ],
            },
        )

    @staticmethod
    def _candidate(
        name: str,
        parameters: dict[str, float],
        ll: float,
        count: int,
        *,
        converged: bool,
        optimizer_message: str,
    ) -> SizeFitCandidate:
        if not math.isfinite(ll):
            raise RuntimeError(f"{name} fit produced a non-finite likelihood")
        parameter_count = len(parameters) + 2  # Lower and upper truncation bounds are estimated from the sample.
        return SizeFitCandidate(
            distribution_type=name,
            parameters=parameters,
            log_likelihood=ll,
            aic=2 * parameter_count - 2 * ll,
            bic=parameter_count * math.log(count) - 2 * ll,
            sample_count=count,
            converged=converged,
            optimizer_message=optimizer_message,
            parameter_count=parameter_count,
        )

    @staticmethod
    def _failed_candidate(name: str, count: int, parameter_count: int, message: str) -> SizeFitCandidate:
        """Record an explicit failed optimizer result without publishing fitted parameters."""
        return SizeFitCandidate(
            distribution_type=name,
            parameters={},
            sample_count=count,
            converged=False,
            optimizer_message=message,
            parameter_count=parameter_count,
        )

    def _uniform(self, values: np.ndarray, lower: float, upper: float) -> SizeFitCandidate:
        ll = -len(values) * math.log(upper - lower)
        return self._candidate(
            "uniform",
            {},
            ll,
            len(values),
            converged=True,
            optimizer_message="closed-form sample-extrema estimate",
        )

    def _lognormal(self, values: np.ndarray, lower: float, upper: float) -> SizeFitCandidate:
        logs = np.log(values)
        mu, sigma = float(logs.mean()), max(float(logs.std(ddof=0)), 1e-6)
        z = stats.norm.cdf((math.log(upper) - mu) / sigma) - stats.norm.cdf((math.log(lower) - mu) / sigma)
        ll = float(np.sum(stats.lognorm.logpdf(values, s=sigma, scale=math.exp(mu))) - len(values) * math.log(z))
        return self._candidate(
            "truncated_lognormal",
            {"mu": mu, "sigma": sigma},
            ll,
            len(values),
            converged=False,
            optimizer_message="EXPERIMENTAL: moment initialization; truncated likelihood not optimized",
        )

    def _power(self, values: np.ndarray, lower: float, upper: float) -> SizeFitCandidate:
        def objective(raw: np.ndarray) -> float:
            exponent = float(np.exp(raw[0]))
            norm = exponent / (lower ** (-exponent) - upper ** (-exponent))
            return -float(len(values) * math.log(norm) - (exponent + 1) * np.log(values).sum())

        result = optimize.minimize(objective, np.array([math.log(2.0)]))
        if not result.success or not np.isfinite(result.fun):
            return self._failed_candidate(
                "truncated_power_law",
                len(values),
                3,
                f"optimization failed: {result.message}",
            )
        exponent = float(np.exp(result.x[0]))
        return self._candidate(
            "truncated_power_law",
            {"exponent": exponent},
            -float(result.fun),
            len(values),
            converged=True,
            optimizer_message=str(result.message),
        )

    def _exponential(self, values: np.ndarray, lower: float, upper: float) -> SizeFitCandidate:
        def objective(raw: np.ndarray) -> float:
            rate = float(np.exp(raw[0]))
            normalizer = math.exp(-rate * lower) - math.exp(-rate * upper)
            return -float(len(values) * math.log(rate / normalizer) - rate * values.sum())

        result = optimize.minimize(objective, np.array([math.log(1 / values.mean())]))
        if not result.success or not np.isfinite(result.fun):
            return self._failed_candidate(
                "truncated_exponential",
                len(values),
                3,
                f"optimization failed: {result.message}",
            )
        rate = float(np.exp(result.x[0]))
        return self._candidate(
            "truncated_exponential",
            {"rate": rate},
            -float(result.fun),
            len(values),
            converged=True,
            optimizer_message=str(result.message),
        )


def assumed_size_model(
    *, domain_id: int | None, set_id: int, distribution_type: str, parameters: dict[str, float], lower: float, upper: float,
    user_defined: bool = True,
) -> SizeModel:
    """Create a clearly labelled manual/assumed model without fake fitting."""
    mean, mean2 = size_moments(distribution_type, parameters, lower, upper)
    return SizeModel(
        domain_id=domain_id,
        set_id=set_id,
        distribution_type=distribution_type,
        parameters=parameters,
        min_radius=lower,
        max_radius=upper,
        mean_radius=mean,
        mean_squared_radius=mean2,
        source=SizeModelSource.USER_DEFINED if user_defined else SizeModelSource.ASSUMED,
        fit_status="assumed" if not user_defined else "user_defined",
        provenance={"automatic_fit": False, "reliability": "ASSUMED" if not user_defined else "USER_DEFINED"},
    )
