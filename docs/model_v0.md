# Model V0: Probabilistic State Estimation from Fuzzy Evidence

## 1. Research question

Can heterogeneous observations be compressed into a continuously updated latent world state that improves calibrated future-event prediction over direct use of raw features?

The general target is

\[
q(z_t\mid o_{\le t}),
\qquad
p(z_{t+1:t+H}\mid z_t,\mathbf a_t),
\qquad
p(y_{t+1:t+H}\mid z_t).
\]

Version 0 does not attempt to solve the full problem. It establishes a precise probabilistic core that later observation encoders can connect to.

## 2. Semantics: fuzzy truth is not probability

Two numbers in `[0, 1]` can mean different things.

### Probability

\[
P(Z_t=s\mid o_{\le t})
\]

represents epistemic uncertainty about which hidden state is true.

### Fuzzy membership

\[
\mu_j(o_t)\in[0,1]
\]

represents the degree to which observation `o_t` satisfies a vague predicate such as “hostile rhetoric,” “strong diplomatic engagement,” or “elevated policy pressure.”

A membership of `0.8` is not automatically an 80% probability that an event occurred. It becomes probabilistic evidence only through an explicit observation model.

## 3. Discrete latent-state filter

Let the hidden state be one of `K` regimes:

\[
Z_t\in\{1,\dots,K\}.
\]

The belief vector is

\[
b_t(k)=P(Z_t=k\mid o_{\le t}).
\]

### Prediction

Given a row-stochastic transition matrix `T`,

\[
\bar b_t(k)=\sum_i b_{t-1}(i)T_{ik}.
\]

An action-conditioned model can later replace `T` with `T(a_t)`.

### Soft evidence update

For each fuzzy predicate `j`, define

\[
\theta_{kj}=P(E_j=1\mid Z_t=k).
\]

Given observed membership `\mu_j` and confidence `c_j`, use a fractional Bernoulli log likelihood:

\[
\log L_k
=
\sum_j c_j
\left[
\mu_j\log\theta_{kj}
+(1-\mu_j)\log(1-\theta_{kj})
\right].
\]

Then update by Bayes' rule:

\[
b_t(k)
=
\frac{\bar b_t(k)L_k}
{\sum_r \bar b_t(r)L_r}.
\]

This is a virtual-evidence interface. A language model, rule system, event extractor, graph encoder, or human annotation process may produce `\mu_j`; the filter remains responsible for probabilistic state uncertainty.

## 4. Why begin with regimes

A discrete state model is not the final architecture. It is the smallest model that makes the following questions testable:

1. Are state and observation uncertainty separated correctly?
2. Does new evidence move beliefs in the expected direction?
3. Does uncertainty propagate through time?
4. Can the system be calibrated and compared with simple baselines?
5. Can different observation encoders share one stable filtering interface?

Starting directly with a large latent neural model would make failure difficult to diagnose.

## 5. Working empirical slice

A reasonable first dataset should fix:

- one geopolitical or policy domain;
- two to four principal actors;
- three to five event classes;
- daily or weekly time steps;
- a strict historical cutoff for every observation;
- rolling out-of-sample forecasting.

A working default is technology-policy competition and cooperation, with event labels such as policy tightening, negotiation, restriction, retaliation, and de-escalation. This is a scope placeholder, not a permanent commitment.

## 6. Baselines

Every complex model must be compared against:

- last-state persistence;
- event-frequency forecasts;
- logistic regression;
- gradient-boosted trees;
- direct text embeddings;
- direct event-count features;
- an end-to-end predictor without an explicit latent state.

## 7. Evaluation

### Event forecasts

- Brier score;
- log loss;
- expected calibration error;
- ranking metrics where appropriate;
- performance by forecast horizon;
- update quality after new evidence.

### State representation

- predictive sufficiency for future events;
- stability under irrelevant paraphrases;
- sensitivity to genuinely new evidence;
- detection of structural breaks;
- separability of similar text with different strategic states.

### Market extension

Market prediction is deferred until the event model is credible. When added, evaluation should focus on conditional return distributions, information coefficients, risk-adjusted performance, drawdown, transaction costs, and regime stability rather than headline directional accuracy.

## 8. Non-goals for V0

Version 0 does not claim to:

- predict global politics;
- infer real leaders' exact utility functions;
- produce trading alpha;
- establish causal identification from observational news;
- validate large-scale LLM social simulation;
- replace a temporal knowledge graph or a full nonlinear filter.

## 9. Extension path

### V1: learned observation model

Replace manually supplied memberships with encoders for text, event tuples, graphs, and numerical time series. Calibrate each encoder against timestamped labels.

### V2: continuous latent dynamics

Use a neural state-space model or SDE:

\[
dz_t=f_\theta(z_t,a_t)dt+\sigma_\theta(z_t)dW_t,
\qquad
do_t=g_\phi(z_t)dt+\eta_t.
\]

Inference may use an extended or unscented Kalman filter, variational filtering, sequential Monte Carlo, or a hybrid multi-scale filter.

### V3: structured actors

Factor the state by actor and relation:

\[
z_t=\{z_t^{(i)},r_t^{(i,j)},m_t\},
\]

where actor states encode goals, resources, beliefs, and readiness; relation states encode cooperation, dependence, deterrence, and conflict; and `m_t` encodes shared macro or market conditions.

### V4: counterfactual rollout

Intervene on actions or policies and estimate scenario branches:

\[
p(z_{t+1:t+H},y_{t+1:t+H}\mid z_t,do(a_t=\tilde a)).
\]

Counterfactual usefulness must be evaluated separately from observational forecast accuracy.

## 10. Immediate milestone

The next empirical milestone is a timestamp-safe table with one row per time step containing:

1. fuzzy predicate values and confidence;
2. observed event labels for future horizons;
3. macro or market covariates available at that timestamp;
4. provenance and publication time;
5. train, validation, and rolling test boundaries.

Only after that table exists should the project add neural encoders or multi-agent simulation.
