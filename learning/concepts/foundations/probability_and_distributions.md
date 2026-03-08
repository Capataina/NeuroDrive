# Probability and Distributions for Reinforcement Learning

## Prerequisites

- Comfortable with exponents and logarithms (ln, e).
- Basic understanding of functions and summation.
- Familiarity with the concept of a function's area under the curve is helpful but not required.

## Target Depth for This Project

**Level 2–3.** You need to understand Gaussian distributions well enough to know what the policy network outputs, how log-probabilities are computed, and why entropy encourages exploration. You do not need measure theory or proofs — but you must be able to compute a log-probability by hand.

---

## Core Concept

### Random variables and probability distributions

A **random variable** is a quantity whose value is determined by a random process. When NeuroDrive's agent chooses a steering angle, that angle is a random variable — it could take many values, each with a different likelihood.

A **probability distribution** describes the likelihood of each possible value. For discrete outcomes (like rolling a die), we assign a probability to each value. For continuous outcomes (like steering angle), we use a **probability density function** (PDF).

**Key property of a PDF:** the total area under the curve equals 1. The probability of the variable falling within a range [a, b] is the area under the PDF between a and b. The PDF value at a single point is **not** a probability — it is a density, and can exceed 1.

### The Gaussian (normal) distribution

The **Gaussian distribution** is the most important continuous distribution in machine learning. Its PDF is:

```
p(x) = (1 / (σ√(2π))) · exp(−(x − μ)² / (2σ²))
```

**Parameters:**

- **μ (mu):** the mean — the centre of the bell curve.
- **σ (sigma):** the standard deviation — controls the width/spread.
- **σ² (sigma squared):** the variance.

**Shape:** a symmetric bell curve centred at μ. Most of the probability mass is concentrated near the mean.

**The 68-95-99.7 rule:**

- 68% of values fall within μ ± 1σ
- 95% of values fall within μ ± 2σ
- 99.7% of values fall within μ ± 3σ

**Concrete example.** If NeuroDrive's steering output has μ = 0.0 and σ = 0.3:

- 68% of sampled steering angles fall between −0.3 and +0.3.
- 95% fall between −0.6 and +0.6.
- Only 0.3% fall outside −0.9 to +0.9.

A small σ means the agent is confident and exploits known behaviour. A large σ means wide exploration.

### Sampling from a Gaussian

To **sample** means to draw a random value from the distribution. The standard method is the **reparameterisation trick**:

```
x = μ + σ · ε,     where ε ~ N(0, 1)
```

First draw ε from a standard normal (mean 0, std 1), then scale by σ and shift by μ. This trick is essential because it makes the sampling operation differentiable with respect to μ and σ — a requirement for training via gradient descent.

---

## Mathematical Foundation

### Log-probability

The **log-probability** is the natural logarithm of the PDF value:

```
log p(x) = ln(p(x))
```

**Why use log-probabilities?**

1. **Numerical stability.** Probabilities can be extremely small (e.g., 10⁻²⁰). Their logarithms are manageable negative numbers (e.g., −46.05).
2. **Products become sums.** The probability of independent events is their product: p(a)·p(b). In log-space: ln(p(a)) + ln(p(b)). Sums are numerically safer and computationally cheaper.
3. **Gradient computation.** Policy gradient methods multiply returns by log-probabilities. Working in log-space avoids underflow.

**Full formula for Gaussian log-probability:**

Starting from the PDF and taking the natural log:

```
log p(x) = −0.5 · [(x − μ)² / σ²  +  ln(2π)  +  2·ln(σ)]
```

This breaks into three terms:

- **(x − μ)² / σ²** — the squared deviation, normalised by variance. Measures how far x is from the mean in units of σ.
- **ln(2π)** — a constant ≈ 1.8379. Always the same regardless of parameters or data.
- **2·ln(σ)** — the normalisation penalty for the spread. Wider distributions (larger σ) have lower peak density.

### Worked example: computing a log-probability

Compute log p(x = 1.5) under N(μ = 1.0, σ = 0.5).

**Step 1:** Compute the squared deviation term.

```
(x − μ)² / σ² = (1.5 − 1.0)² / 0.5²
              = (0.5)² / (0.25)
              = 0.25 / 0.25
              = 1.0
```

**Step 2:** Compute the constant term.

```
ln(2π) = ln(6.2832) ≈ 1.8379
```

**Step 3:** Compute the spread term.

```
2 · ln(σ) = 2 · ln(0.5) = 2 · (−0.6931) = −1.3863
```

**Step 4:** Combine.

```
log p(x) = −0.5 · [1.0 + 1.8379 + (−1.3863)]
         = −0.5 · [1.4516]
         = −0.7258
```

So log p(1.5) ≈ −0.7258.

**Verification:** exp(−0.7258) ≈ 0.4839. Using the PDF directly: (1/(0.5 × √(2π))) · exp(−1.0/2) = (1/1.2533) · 0.6065 ≈ 0.7979 · 0.6065 ≈ 0.4839 ✓.

### Entropy of a Gaussian

**Entropy** measures the uncertainty or spread of a distribution. For a Gaussian:

```
H = 0.5 + 0.5·ln(2π) + ln(σ)
```

This simplifies to: H ≈ 1.4189 + ln(σ).

**Key insight:** entropy depends **only on σ**, not on μ. Shifting the centre of the bell curve does not change how "spread out" it is.

**Numerical examples:**

- σ = 0.1: H ≈ 1.4189 + (−2.3026) = −0.8837. Low entropy — the distribution is very narrow and peaked. Little exploration.
- σ = 0.5: H ≈ 1.4189 + (−0.6931) = 0.7258. Moderate entropy.
- σ = 1.0: H ≈ 1.4189 + 0.0 = 1.4189. Baseline entropy of a standard normal.
- σ = 2.0: H ≈ 1.4189 + 0.6931 = 2.1120. High entropy — wide, uncertain distribution. Maximum exploration.

Higher entropy → wider distribution → the agent explores more diverse actions.

---

## How NeuroDrive Uses This

**Gaussian policy.** NeuroDrive's actor network outputs two values per action dimension: μ (the mean) and σ (the standard deviation). Together, they define a Gaussian distribution over possible actions (steering, throttle). The agent **samples** from this distribution each time it acts.

**Log-probability for policy gradients.** The A2C algorithm computes the log-probability of the action that was actually taken, using exactly the formula above. This log-prob is multiplied by the advantage (how much better the action was than expected) to form the policy gradient loss. Actions with high advantage and high log-prob receive the strongest reinforcement.

**Entropy bonus for exploration.** NeuroDrive adds an entropy bonus to the loss: L_entropy = −β · H, where β is the entropy coefficient. This penalises low-entropy (overly confident) policies and encourages the agent to maintain σ large enough for meaningful exploration — especially critical in early training when the agent has not yet discovered which actions are good.

**σ as a training signal.** Monitoring how σ evolves over training reveals whether the agent is exploring (σ stays large) or exploiting (σ shrinks). If σ collapses to near zero too early, the agent has become prematurely deterministic and may be stuck in a poor local optimum.

---

## Common Misconceptions

1. **"The PDF value at a point is the probability of that exact value."** For continuous distributions, the probability of any single exact value is technically zero. The PDF gives a **density** — you need to integrate over a range to get a probability. A density can exceed 1 (e.g., N(0, 0.1) has a peak density of about 3.99).

2. **"Higher log-probability means the action was better."** Higher log-probability means the action was **more likely under the current policy**, not that it was a good action. An agent can assign high probability to a terrible action. The advantage term in the policy gradient is what distinguishes good from bad.

3. **"Entropy should always be maximised."** Entropy is a regulariser, not the objective. Too much entropy means the agent acts randomly and cannot exploit what it has learnt. The goal is a balance: enough entropy to explore, not so much that behaviour is noise.

---

## Glossary

| Term | Definition |
|---|---|
| **Random variable** | A quantity whose value is determined by a random process. |
| **PDF (probability density function)** | A function whose integral over a range gives the probability of falling in that range. |
| **Gaussian / normal distribution** | A bell-shaped distribution parameterised by mean μ and standard deviation σ. |
| **Mean (μ)** | The centre of the distribution; the expected value. |
| **Standard deviation (σ)** | Measures the spread of the distribution. |
| **Log-probability** | The natural logarithm of the PDF value; used for numerical stability. |
| **Entropy** | A measure of the uncertainty/spread of a distribution. |
| **Reparameterisation trick** | x = μ + σε; makes sampling differentiable. |

---

## Recommended Materials

1. **StatQuest — "The Normal Distribution, Clearly Explained"** (YouTube). An accessible, visual introduction to Gaussian distributions, their parameters, and the 68-95-99.7 rule.
2. **David Silver — "Reinforcement Learning Lecture 7: Policy Gradient Methods"** (YouTube / UCL). Covers how log-probabilities and entropy appear in policy gradient algorithms — the exact context NeuroDrive uses.
3. **Mathematics for Machine Learning** — Deisenroth, Faisal & Ong, Chapter 6. Covers probability distributions, Gaussians, and their properties with ML-oriented examples and exercises.
