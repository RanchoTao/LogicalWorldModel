import numpy as np
import pytest

from logical_world_model import FuzzyBayesFilter


def make_filter() -> FuzzyBayesFilter:
    return FuzzyBayesFilter(
        transition=[
            [0.85, 0.15],
            [0.20, 0.80],
        ],
        predicate_likelihoods=[
            [0.15, 0.80],
            [0.85, 0.20],
        ],
        prior=[0.50, 0.50],
        state_names=["low_pressure", "high_pressure"],
        predicate_names=["restriction", "diplomacy"],
    )


def test_probabilities_remain_normalized() -> None:
    model = make_filter()
    posterior = model.step([0.75, 0.20], confidence=[0.9, 0.8])
    assert posterior.shape == (2,)
    assert np.isclose(posterior.sum(), 1.0)
    assert np.all(posterior >= 0)


def test_restrictive_evidence_increases_high_pressure_belief() -> None:
    model = make_filter()
    predicted = model.predict()
    posterior = model.update([0.95, 0.05], confidence=[1.0, 1.0])
    assert posterior[1] > predicted[1]


def test_zero_confidence_leaves_predicted_belief_unchanged() -> None:
    model = make_filter()
    predicted = model.predict()
    posterior = model.update([1.0, 0.0], confidence=[0.0, 0.0])
    assert np.allclose(posterior, predicted)


def test_forecast_does_not_mutate_current_belief() -> None:
    model = make_filter()
    model.step([0.70, 0.30])
    current = model.belief.copy()
    forecast = model.forecast(4)
    assert forecast.shape == (4, 2)
    assert np.allclose(forecast.sum(axis=1), 1.0)
    assert np.allclose(model.belief, current)


def test_invalid_membership_is_rejected() -> None:
    model = make_filter()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        model.update([1.2, 0.4])


def test_update_sequence_checks_lengths() -> None:
    model = make_filter()
    with pytest.raises(ValueError, match="match memberships length"):
        model.update_sequence(
            memberships=[[0.2, 0.8], [0.7, 0.3]],
            confidences=[[1.0, 1.0]],
        )
