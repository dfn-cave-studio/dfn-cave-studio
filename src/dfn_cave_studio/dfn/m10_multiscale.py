"""Deterministic multiscale radius budgets for M10 explicit DFN generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum
import math

import numpy as np
from scipy import optimize, stats

from dfn_cave_studio.models.m9 import SizeModel


THRESHOLD_METHOD_VERSION = "area-weighted-cdf-1"


class SizeClass(IntEnum):
    """Compact explicit-fracture size class codes."""

    UNKNOWN = 0
    SMALL = 1
    MEDIUM = 2
    LARGE = 3


class ThresholdMode(StrEnum):
    """Supported size-threshold modes."""

    AUTO = "auto"
    MANUAL = "manual"


@dataclass(frozen=True)
class SizeClassBudget:
    """Probability and area-weighted budget for one mutually exclusive class."""

    size_class: SizeClass
    lower: float
    upper: float
    probability: float
    area_share: float
    conditional_mean_squared_radius: float | None


@dataclass(frozen=True)
class SizeThresholdResult:
    """Thresholds and complete probability/P32 partition for one size model."""

    mode: ThresholdMode
    small_medium_radius: float
    medium_large_radius: float
    budgets: tuple[SizeClassBudget, SizeClassBudget, SizeClassBudget]
    method_version: str = THRESHOLD_METHOD_VERSION
    notice: str = "Thresholds are numerical modelling resolution recommendations, not fixed geological classes."
    degenerate: bool = False


def _validate_model(model: SizeModel) -> tuple[float, float]:
    lower, upper = float(model.min_radius), float(model.max_radius)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0.0 or upper < lower:
        raise ValueError("Size distribution requires finite bounds with 0 < min_radius <= max_radius")
    if not math.isfinite(float(model.mean_squared_radius)) or model.mean_squared_radius <= 0.0:
        raise ValueError("Size distribution does not provide a positive finite E[R^2]")
    return lower, upper


def probability_cdf(model: SizeModel, radius: float) -> float:
    """Return deterministic P(R <= radius) for every supported distribution."""
    lower, upper = _validate_model(model)
    x = float(radius)
    if x < lower:
        return 0.0
    if x >= upper:
        return 1.0
    if model.distribution_type == "fixed":
        fixed = float(model.parameters.get("radius", lower))
        return 0.0 if x < fixed else 1.0
    if model.distribution_type == "uniform":
        return (x - lower) / (upper - lower)
    if model.distribution_type == "truncated_lognormal":
        mu, sigma = float(model.parameters["mu"]), float(model.parameters["sigma"])
        if sigma <= 0.0:
            raise ValueError("truncated_lognormal requires sigma > 0")
        lo = stats.norm.cdf((math.log(lower) - mu) / sigma)
        hi = stats.norm.cdf((math.log(upper) - mu) / sigma)
        return float((stats.norm.cdf((math.log(x) - mu) / sigma) - lo) / (hi - lo))
    if model.distribution_type == "truncated_power_law":
        exponent = float(model.parameters["exponent"])
        if exponent <= 0.0:
            raise ValueError("truncated_power_law requires exponent > 0")
        return (lower ** (-exponent) - x ** (-exponent)) / (
            lower ** (-exponent) - upper ** (-exponent)
        )
    if model.distribution_type == "truncated_exponential":
        rate = float(model.parameters["rate"])
        if rate <= 0.0:
            raise ValueError("truncated_exponential requires rate > 0")
        return (math.exp(-rate * lower) - math.exp(-rate * x)) / (
            math.exp(-rate * lower) - math.exp(-rate * upper)
        )
    raise ValueError(f"Unsupported M10 size distribution: {model.distribution_type}")


def area_weighted_cdf(model: SizeModel, radius: float) -> float:
    """Return F_A(r)=E[R² I(R<=r)]/E[R²] without stochastic sampling."""
    lower, upper = _validate_model(model)
    x = float(radius)
    if x < lower:
        return 0.0
    if x >= upper:
        return 1.0
    if model.distribution_type == "fixed":
        fixed = float(model.parameters.get("radius", lower))
        return 0.0 if x < fixed else 1.0
    if model.distribution_type == "uniform":
        return (x**3 - lower**3) / (upper**3 - lower**3)
    if model.distribution_type == "truncated_lognormal":
        mu, sigma = float(model.parameters["mu"]), float(model.parameters["sigma"])
        shifted_lower = (math.log(lower) - mu - 2.0 * sigma * sigma) / sigma
        shifted_upper = (math.log(upper) - mu - 2.0 * sigma * sigma) / sigma
        shifted_x = (math.log(x) - mu - 2.0 * sigma * sigma) / sigma
        return float(
            (stats.norm.cdf(shifted_x) - stats.norm.cdf(shifted_lower))
            / (stats.norm.cdf(shifted_upper) - stats.norm.cdf(shifted_lower))
        )
    if model.distribution_type == "truncated_power_law":
        exponent = float(model.parameters["exponent"])
        delta = 2.0 - exponent
        if abs(delta) < 1e-12:
            return math.log(x / lower) / math.log(upper / lower)
        return (x**delta - lower**delta) / (upper**delta - lower**delta)
    if model.distribution_type == "truncated_exponential":
        rate = float(model.parameters["rate"])

        def antiderivative(value: float) -> float:
            return -math.exp(-rate * value) * (
                value * value + 2.0 * value / rate + 2.0 / (rate * rate)
            )

        return (antiderivative(x) - antiderivative(lower)) / (
            antiderivative(upper) - antiderivative(lower)
        )
    raise ValueError(f"Unsupported M10 size distribution: {model.distribution_type}")


def _inverse_area_cdf(model: SizeModel, target: float) -> float:
    lower, upper = _validate_model(model)
    if not 0.0 < target < 1.0:
        raise ValueError("Area-weighted cumulative target must be between zero and one")
    return float(optimize.brentq(lambda value: area_weighted_cdf(model, value) - target, lower, upper))


def validate_manual_thresholds(small_medium_radius: float, medium_large_radius: float) -> None:
    """Reject invalid manual thresholds explicitly."""
    first, second = float(small_medium_radius), float(medium_large_radius)
    if not math.isfinite(first) or not math.isfinite(second) or first < 0.0 or first >= second:
        raise ValueError("Manual thresholds require 0 <= r_sm < r_ml")


def calculate_thresholds(
    model: SizeModel,
    *,
    mode: ThresholdMode | str = ThresholdMode.AUTO,
    small_area_share: float = 0.10,
    medium_large_cumulative_share: float = 0.70,
    manual_small_medium_radius: float = 0.5,
    manual_medium_large_radius: float = 2.0,
) -> SizeThresholdResult:
    """Calculate thresholds and a complete mutually exclusive distribution partition."""
    threshold_mode = ThresholdMode(mode)
    lower, upper = _validate_model(model)
    if threshold_mode == ThresholdMode.AUTO:
        if not 0.0 < small_area_share < medium_large_cumulative_share < 1.0:
            raise ValueError("Auto shares require 0 < small share < medium/large cumulative share < 1")
        if model.distribution_type == "fixed" or math.isclose(lower, upper):
            fixed = float(model.parameters.get("radius", lower))
            r_sm = max(0.0, float(np.nextafter(fixed, -math.inf)))
            r_ml = fixed
            degenerate = True
        else:
            r_sm = _inverse_area_cdf(model, small_area_share)
            r_ml = _inverse_area_cdf(model, medium_large_cumulative_share)
            degenerate = False
    else:
        validate_manual_thresholds(manual_small_medium_radius, manual_medium_large_radius)
        r_sm, r_ml = float(manual_small_medium_radius), float(manual_medium_large_radius)
        degenerate = model.distribution_type == "fixed" or math.isclose(lower, upper)

    if degenerate:
        # Allocate a point mass according to the documented half-open ranges.
        fixed = float(model.parameters.get("radius", lower))
        if fixed < r_sm:
            probability_edges = area_edges = (0.0, 1.0, 1.0, 1.0)
        elif fixed < r_ml:
            probability_edges = area_edges = (0.0, 0.0, 1.0, 1.0)
        else:
            probability_edges = area_edges = (0.0, 0.0, 0.0, 1.0)
    else:
        probability_edges = (0.0, probability_cdf(model, r_sm), probability_cdf(model, r_ml), 1.0)
        area_edges = (0.0, area_weighted_cdf(model, r_sm), area_weighted_cdf(model, r_ml), 1.0)
    ranges = ((0.0, r_sm), (r_sm, r_ml), (r_ml, math.inf))
    classes = (SizeClass.SMALL, SizeClass.MEDIUM, SizeClass.LARGE)
    budgets: list[SizeClassBudget] = []
    for index, size_class in enumerate(classes):
        probability = max(0.0, probability_edges[index + 1] - probability_edges[index])
        area_share = max(0.0, area_edges[index + 1] - area_edges[index])
        conditional_second = (
            float(model.mean_squared_radius) * area_share / probability if probability > 0.0 else None
        )
        budgets.append(
            SizeClassBudget(
                size_class=size_class,
                lower=ranges[index][0],
                upper=ranges[index][1],
                probability=probability,
                area_share=area_share,
                conditional_mean_squared_radius=conditional_second,
            )
        )
    probability_error = abs(sum(item.probability for item in budgets) - 1.0)
    area_error = abs(sum(item.area_share for item in budgets) - 1.0)
    if probability_error > 1e-10 or area_error > 1e-10:
        raise ArithmeticError("Size-class distribution partition failed conservation")
    return SizeThresholdResult(
        mode=threshold_mode,
        small_medium_radius=r_sm,
        medium_large_radius=r_ml,
        budgets=tuple(budgets),
        degenerate=degenerate,
    )


def classify_radius(radius: float, thresholds: SizeThresholdResult) -> SizeClass:
    """Classify one radius using the documented half-open ranges."""
    value = float(radius)
    if value < thresholds.small_medium_radius:
        return SizeClass.SMALL
    if value < thresholds.medium_large_radius:
        return SizeClass.MEDIUM
    return SizeClass.LARGE


def sample_truncated_class(
    model: SizeModel,
    budget: SizeClassBudget,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample directly from one conditional size class using inverse CDF."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return np.empty(0, dtype=np.float64)
    if budget.probability <= 0.0:
        raise ValueError(f"Cannot sample empty {budget.size_class.name} radius class")
    model_lower, model_upper = _validate_model(model)
    lower = max(model_lower, budget.lower)
    upper = min(model_upper, budget.upper)
    if model.distribution_type == "fixed":
        radius = float(model.parameters.get("radius", model_lower))
        in_range = radius >= budget.lower and (not math.isfinite(budget.upper) or radius < budget.upper)
        if not in_range:
            raise ValueError("Fixed radius is outside the requested size class")
        return np.full(count, radius, dtype=np.float64)
    cdf_lower = probability_cdf(model, float(np.nextafter(lower, -math.inf)))
    cdf_upper = probability_cdf(model, upper)
    u = cdf_lower + rng.random(count) * (cdf_upper - cdf_lower)
    if model.distribution_type == "uniform":
        values = model_lower + u * (model_upper - model_lower)
    elif model.distribution_type == "truncated_lognormal":
        mu, sigma = float(model.parameters["mu"]), float(model.parameters["sigma"])
        lo = stats.norm.cdf((math.log(model_lower) - mu) / sigma)
        hi = stats.norm.cdf((math.log(model_upper) - mu) / sigma)
        values = np.exp(mu + sigma * stats.norm.ppf(lo + u * (hi - lo)))
    elif model.distribution_type == "truncated_power_law":
        exponent = float(model.parameters["exponent"])
        values = (
            model_lower ** (-exponent)
            - u * (model_lower ** (-exponent) - model_upper ** (-exponent))
        ) ** (-1.0 / exponent)
    elif model.distribution_type == "truncated_exponential":
        rate = float(model.parameters["rate"])
        values = -np.log(
            math.exp(-rate * model_lower)
            - u * (math.exp(-rate * model_lower) - math.exp(-rate * model_upper))
        ) / rate
    else:
        raise ValueError(f"Unsupported M10 size distribution: {model.distribution_type}")
    return np.clip(np.asarray(values, dtype=np.float64), lower, upper)
