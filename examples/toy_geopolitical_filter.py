"""Toy sequential update for a three-regime geopolitical state model.

The numbers are illustrative only. They demonstrate the interface and should not
be interpreted as empirical estimates of any real country or event.
"""

from logical_world_model import FuzzyBayesFilter


filter_model = FuzzyBayesFilter(
    transition=[
        [0.75, 0.22, 0.03],
        [0.10, 0.78, 0.12],
        [0.03, 0.27, 0.70],
    ],
    predicate_likelihoods=[
        # hostile rhetoric, policy restriction, diplomacy, military signal
        [0.15, 0.10, 0.85, 0.08],
        [0.55, 0.50, 0.45, 0.35],
        [0.88, 0.82, 0.18, 0.78],
    ],
    prior=[0.25, 0.60, 0.15],
    state_names=["de_escalation", "managed_competition", "escalation"],
    predicate_names=[
        "hostile_rhetoric",
        "policy_restriction",
        "diplomatic_engagement",
        "military_signal",
    ],
)

observations = [
    {
        "label": "routine competitive statements",
        "membership": [0.55, 0.40, 0.45, 0.20],
        "confidence": [0.80, 0.70, 0.60, 0.50],
    },
    {
        "label": "new restriction with limited diplomatic contact",
        "membership": [0.72, 0.90, 0.18, 0.30],
        "confidence": [0.85, 0.95, 0.75, 0.50],
    },
    {
        "label": "credible negotiation signal",
        "membership": [0.28, 0.35, 0.88, 0.12],
        "confidence": [0.70, 0.75, 0.90, 0.45],
    },
]

print("initial", filter_model.as_dict())
for observation in observations:
    filter_model.step(
        observation["membership"],
        confidence=observation["confidence"],
    )
    print(observation["label"], filter_model.as_dict())

print("three-step forecast")
for horizon, probabilities in enumerate(filter_model.forecast(3), start=1):
    print(
        horizon,
        dict(zip(filter_model.state_names, probabilities, strict=True)),
    )
