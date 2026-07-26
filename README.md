# Logical World Model

A research codebase for **probabilistic world-state estimation with fuzzy evidence**.

The project studies whether heterogeneous public observations can be compressed into an online-updated latent state that is more useful for forecasting than raw text embeddings, event counts, or direct end-to-end prediction.

## Core problem

Given observations up to time `t`, infer a hidden world state and forecast its future consequences:

```text
o_{<=t}
  -> q(z_t | o_{<=t})
  -> p(z_{t+1:t+H} | z_t, a_t)
  -> p(events, macro outcomes, market reactions | z_t)
```

The latent state should eventually represent structured quantities such as actor goals, constraints, relationships, beliefs, action readiness, and market expectations. It must remain uncertain, update when new evidence arrives, and support counterfactual intervention.

## Important distinction

This repository does **not** treat fuzzy truth and probability as the same object.

- Probability describes uncertainty about a proposition or hidden state.
- Fuzzy membership describes how strongly a vague predicate applies to an observation.

Version 0 converts fuzzy memberships into calibrated soft evidence, then performs Bayesian filtering over latent regimes.

## Version 0

The first executable model is a small discrete-state filter:

```text
fuzzy observations
  -> virtual evidence likelihood
  -> Bayesian state update
  -> multi-step state forecast
```

It is deliberately small. The goal is to validate the semantics, interfaces, tests, and evaluation protocol before adding language models, temporal graphs, neural state-space models, SDEs, particle filters, or multi-agent simulation.

## Quick start

```bash
python -m pip install -e .
python examples/toy_geopolitical_filter.py
python -m pytest
```

## Research phases

1. **Latent state representation and calibrated event forecasting**
2. **Actor beliefs, actions, and counterfactual rollouts**
3. **Macroeconomic transmission and market repricing**

The working empirical slice for the first paper is intentionally narrow: a limited region, a few actors, several event classes, daily or weekly observations, and strict rolling out-of-sample evaluation.

## Repository layout

```text
docs/model_v0.md                 mathematical and research specification
src/logical_world_model/filter.py executable fuzzy Bayesian filter
examples/toy_geopolitical_filter.py minimal sequential update example
tests/test_filter.py              behavioral tests
Thoughts/                         original project notes
```

## Current research criterion

The first publishable claim is not “predict the world.” It is:

> A continuously updated latent state built from heterogeneous evidence improves calibrated future-event prediction over simple direct-feature baselines.

See [`docs/model_v0.md`](docs/model_v0.md) for the exact semantics, equations, non-goals, evaluation plan, and extension path.
