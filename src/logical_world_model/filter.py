"""A small Bayesian filter that consumes fuzzy observations as soft evidence.

Probability and fuzzy membership are intentionally kept distinct:

- ``belief[k]`` is a probability over mutually exclusive latent states.
- ``membership[j]`` is the degree to which a vague observation predicate applies.

The observation model maps memberships to probabilistic virtual evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _as_probability_vector(values: ArrayLike, *, name: str) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if vector.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    if np.any(vector < 0):
        raise ValueError(f"{name} cannot contain negative values")
    total = float(vector.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return vector / total


def _as_unit_interval_vector(
    values: ArrayLike,
    *,
    name: str,
    expected_size: int,
) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.shape != (expected_size,):
        raise ValueError(f"{name} must have shape ({expected_size},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain finite values")
    if np.any((vector < 0) | (vector > 1)):
        raise ValueError(f"{name} values must lie in [0, 1]")
    return vector


@dataclass
class FuzzyBayesFilter:
    """Discrete latent-state Bayesian filter with fuzzy virtual evidence.

    Parameters
    ----------
    transition:
        Row-stochastic matrix where ``transition[i, j]`` is
        ``P(Z_t=j | Z_{t-1}=i)``.
    predicate_likelihoods:
        Matrix where ``predicate_likelihoods[k, j]`` is
        ``P(E_j=1 | Z_t=k)`` for latent state ``k`` and fuzzy predicate ``j``.
    prior:
        Initial probability distribution over latent states.
    state_names:
        Optional human-readable state labels.
    predicate_names:
        Optional human-readable predicate labels.
    epsilon:
        Numerical clipping value for likelihoods.
    """

    transition: ArrayLike
    predicate_likelihoods: ArrayLike
    prior: ArrayLike
    state_names: Sequence[str] | None = None
    predicate_names: Sequence[str] | None = None
    epsilon: float = 1e-9

    def __post_init__(self) -> None:
        transition = np.asarray(self.transition, dtype=float)
        likelihoods = np.asarray(self.predicate_likelihoods, dtype=float)

        if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
            raise ValueError("transition must be a square matrix")
        if not np.all(np.isfinite(transition)) or np.any(transition < 0):
            raise ValueError("transition must contain finite non-negative values")
        row_sums = transition.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("every transition row must have positive mass")
        transition = transition / row_sums

        n_states = transition.shape[0]
        if likelihoods.ndim != 2 or likelihoods.shape[0] != n_states:
            raise ValueError(
                "predicate_likelihoods must have shape "
                "(number_of_states, number_of_predicates)"
            )
        if likelihoods.shape[1] == 0:
            raise ValueError("at least one predicate is required")
        if not np.all(np.isfinite(likelihoods)):
            raise ValueError("predicate_likelihoods must contain finite values")
        if np.any((likelihoods <= 0) | (likelihoods >= 1)):
            raise ValueError(
                "predicate_likelihoods must lie strictly between 0 and 1"
            )

        prior = _as_probability_vector(self.prior, name="prior")
        if prior.shape != (n_states,):
            raise ValueError(f"prior must have shape ({n_states},)")

        if not 0 < self.epsilon < 0.5:
            raise ValueError("epsilon must lie in (0, 0.5)")

        n_predicates = likelihoods.shape[1]
        if self.state_names is None:
            self.state_names = tuple(f"state_{i}" for i in range(n_states))
        elif len(self.state_names) != n_states:
            raise ValueError("state_names length must match number of states")
        else:
            self.state_names = tuple(self.state_names)

        if self.predicate_names is None:
            self.predicate_names = tuple(
                f"predicate_{j}" for j in range(n_predicates)
            )
        elif len(self.predicate_names) != n_predicates:
            raise ValueError(
                "predicate_names length must match number of predicates"
            )
        else:
            self.predicate_names = tuple(self.predicate_names)

        self.transition = transition
        self.predicate_likelihoods = likelihoods
        self.prior = prior
        self.belief = prior.copy()

    @property
    def n_states(self) -> int:
        return int(self.transition.shape[0])

    @property
    def n_predicates(self) -> int:
        return int(self.predicate_likelihoods.shape[1])

    def reset(self, prior: ArrayLike | None = None) -> FloatArray:
        """Reset the posterior to the original or a supplied prior."""
        if prior is None:
            self.belief = self.prior.copy()
        else:
            candidate = _as_probability_vector(prior, name="prior")
            if candidate.shape != (self.n_states,):
                raise ValueError(f"prior must have shape ({self.n_states},)")
            self.belief = candidate
        return self.belief.copy()

    def predict(self, transition: ArrayLike | None = None) -> FloatArray:
        """Propagate the current state belief through one transition step."""
        matrix = self.transition if transition is None else np.asarray(transition, dtype=float)
        if matrix.shape != (self.n_states, self.n_states):
            raise ValueError(
                f"transition must have shape ({self.n_states}, {self.n_states})"
            )
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError("transition must contain finite non-negative values")
        row_sums = matrix.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("every transition row must have positive mass")
        matrix = matrix / row_sums
        self.belief = self.belief @ matrix
        self.belief = self.belief / self.belief.sum()
        return self.belief.copy()

    def evidence_log_likelihood(
        self,
        memberships: ArrayLike,
        confidence: ArrayLike | None = None,
    ) -> FloatArray:
        """Return per-state log likelihood induced by fuzzy evidence.

        A confidence of zero removes a predicate from the update. Fractional
        memberships use the expected Bernoulli log likelihood and therefore do
        not masquerade as posterior probabilities.
        """
        memberships_vector = _as_unit_interval_vector(
            memberships,
            name="memberships",
            expected_size=self.n_predicates,
        )
        if confidence is None:
            confidence_vector = np.ones(self.n_predicates, dtype=float)
        else:
            confidence_vector = _as_unit_interval_vector(
                confidence,
                name="confidence",
                expected_size=self.n_predicates,
            )

        theta = np.clip(
            self.predicate_likelihoods,
            self.epsilon,
            1 - self.epsilon,
        )
        positive = memberships_vector * np.log(theta)
        negative = (1 - memberships_vector) * np.log(1 - theta)
        return ((positive + negative) * confidence_vector).sum(axis=1)

    def update(
        self,
        memberships: ArrayLike,
        confidence: ArrayLike | None = None,
    ) -> FloatArray:
        """Condition the predicted belief on fuzzy observations."""
        log_likelihood = self.evidence_log_likelihood(memberships, confidence)
        log_weights = np.log(np.clip(self.belief, self.epsilon, 1.0)) + log_likelihood
        log_weights -= np.max(log_weights)
        weights = np.exp(log_weights)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("posterior normalization failed")
        self.belief = weights / total
        return self.belief.copy()

    def step(
        self,
        memberships: ArrayLike,
        confidence: ArrayLike | None = None,
        transition: ArrayLike | None = None,
    ) -> FloatArray:
        """Run one prediction-update cycle."""
        self.predict(transition=transition)
        return self.update(memberships=memberships, confidence=confidence)

    def forecast(self, horizon: int) -> FloatArray:
        """Forecast state probabilities without mutating the current belief."""
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        forecasts = np.empty((horizon, self.n_states), dtype=float)
        belief = self.belief.copy()
        for index in range(horizon):
            belief = belief @ self.transition
            belief = belief / belief.sum()
            forecasts[index] = belief
        return forecasts

    def entropy(self) -> float:
        """Return posterior Shannon entropy in nats."""
        belief = np.clip(self.belief, self.epsilon, 1.0)
        return float(-np.sum(belief * np.log(belief)))

    def as_dict(self) -> dict[str, float]:
        """Return the current state belief keyed by state name."""
        return {
            name: float(probability)
            for name, probability in zip(self.state_names, self.belief, strict=True)
        }

    def update_sequence(
        self,
        memberships: Iterable[ArrayLike],
        confidences: Iterable[ArrayLike | None] | None = None,
    ) -> list[FloatArray]:
        """Apply multiple prediction-update cycles and return their posteriors."""
        membership_rows = list(memberships)
        if confidences is None:
            confidence_rows: list[ArrayLike | None] = [None] * len(membership_rows)
        else:
            confidence_rows = list(confidences)
            if len(confidence_rows) != len(membership_rows):
                raise ValueError("confidences must match memberships length")
        return [
            self.step(row, confidence)
            for row, confidence in zip(
                membership_rows,
                confidence_rows,
                strict=True,
            )
        ]
