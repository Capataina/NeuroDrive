# Materials: Neuroscience-Inspired Learning

Resources for understanding the biological learning mechanisms that NeuroDrive aims to implement beyond the A2C baseline. These cover Hebbian plasticity, spike-timing-dependent plasticity (STDP), eligibility traces, neuromodulation, and three-factor learning rules — the theoretical foundation for NeuroDrive's planned Milestones 2–5.

---

## Hebbian Learning

- [ ] **Dayan & Abbott — *Theoretical Neuroscience*, Chapter 8: "Plasticity and Learning"**
  Link: Available via MIT Press or institutional access.
  Section: Sections 8.1–8.3 (pp. 259–285). Focus on 8.1 "Synaptic Plasticity" (Hebb's postulate, rate-based models) and 8.2 "Unsupervised Learning" (covariance rule, BCM theory, Oja's rule).
  Why: The canonical treatment of synaptic plasticity from a computational neuroscience perspective. The progression from Hebb's postulate ("fire together, wire together") to formal learning rules (Δw = η · x · y) is derived carefully. Understanding the instability problem (unbounded weight growth without normalisation) is essential for NeuroDrive's implementation.
  Difficulty: Advanced | Time: 4–5 hours

- [ ] **Gerstner et al. — *Neuronal Dynamics*, Chapter 19: "Hebbian Models"**
  Link: https://neuronaldynamics.epfl.ch/online/Ch19.html
  Section: Sections 19.1–19.3. Free online textbook.
  Why: A more accessible treatment than Dayan & Abbott, with interactive simulations. The derivation of the BCM rule (which introduces a sliding threshold for LTP/LTD) is particularly clear and relevant to NeuroDrive's planned homeostatic mechanisms.
  Difficulty: Intermediate–Advanced | Time: 2–3 hours

- [ ] **Sejnowski & Tesauro (1989) — "The Hebb Rule for Synaptic Plasticity: Algorithms and Implementations"**
  Link: Available via academic search (Springer).
  Section: Sections 1–3.
  Why: A concise review that connects Hebb's biological principle to computational learning algorithms. Shows how Hebbian learning relates to principal component analysis — useful for understanding what unsupervised Hebbian networks actually learn.
  Difficulty: Advanced | Time: 1.5 hours

---

## Spike-Timing-Dependent Plasticity (STDP)

- [ ] **Bi & Poo (1998) — "Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type"**
  Link: https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998
  Section: Full paper (8 pages). Focus on Figures 5–7 showing the STDP timing window.
  Why: The landmark experimental paper that established the STDP timing rule: pre-before-post → LTP, post-before-pre → LTD. The timing window (±20ms) and the asymmetric exponential fit are the basis for all computational STDP models, including the rate-based approximation NeuroDrive will use.
  Difficulty: Intermediate (experimental, not heavy maths) | Time: 1.5 hours

- [ ] **Markram et al. (2012) — "A History of Spike-Timing-Dependent Plasticity"**
  Link: https://doi.org/10.3389/fnsyn.2012.00002
  Section: Full review (18 pages). Focus on Sections 2–4 covering the experimental evidence, computational models, and functional consequences.
  Why: A comprehensive review from one of the field's founders. Places STDP in historical context (from Hebb through to modern three-factor rules) and discusses how STDP interacts with neuromodulation — exactly the bridge NeuroDrive needs between local plasticity and reward signals.
  Difficulty: Intermediate–Advanced | Time: 2–3 hours

- [ ] **Dan & Poo (2004) — "Spike Timing-Dependent Plasticity of Neural Circuits"**
  Link: https://doi.org/10.1016/j.neuron.2004.09.007
  Section: Full review (Neuron, 12 pages). Focus on "Computational Implications" section.
  Why: Bridges experimental STDP findings to computational consequences. Explains how STDP can implement temporal sequence learning and input correlation detection — relevant to NeuroDrive's sensor-to-action timing.
  Difficulty: Intermediate | Time: 1.5 hours

---

## Neuromodulation and Dopamine

- [ ] **Schultz (1997) — "A Neural Substrate of Prediction and Reward"**
  Link: https://doi.org/10.1126/science.275.5306.1593
  Section: Full paper (Science, 4 pages). Focus on Figures 1–3 showing dopamine neuron firing patterns.
  Why: The foundational paper establishing that dopamine neurons encode reward prediction error (RPE): they fire for unexpected rewards, are suppressed for unexpected reward omissions, and shift their response to reward-predictive cues. This is the biological analogue of the TD error δ_t that NeuroDrive uses for advantage estimation — and the planned neuromodulatory signal for gating plasticity.
  Difficulty: Intermediate | Time: 1 hour

- [ ] **Izhikevich (2007) — "Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling"**
  Link: https://doi.org/10.1093/cercor/bhl152
  Section: Full paper (Cerebral Cortex, 10 pages). Focus on the model description (Section 2) and Figure 1 showing how eligibility traces bridge the temporal gap.
  Why: Directly addresses NeuroDrive's core challenge: how can local synaptic plasticity (which operates on millisecond timescales) be guided by reward signals (which arrive seconds later)? The answer — eligibility traces that maintain a decaying record of recent synaptic activity, "stamped in" by dopamine — is the architecture NeuroDrive plans to implement.
  Difficulty: Advanced | Time: 2–3 hours

- [ ] **Schultz, Dayan & Montague (1997) — "A Neural Substrate of Prediction and Reward"**
  Link: https://doi.org/10.1126/science.275.5306.1593
  Section: Companion paper to Schultz (1997). Focus on the TD learning model of dopamine.
  Why: Formalises the connection between biological dopamine signalling and the temporal-difference learning algorithm. The parallel between δ_t = r + γV(s') − V(s) and dopamine RPE is not a metaphor — it is a quantitative correspondence. Essential reading for understanding why NeuroDrive's transition from A2C to neuromodulated plasticity is scientifically motivated.
  Difficulty: Advanced | Time: 1.5 hours

- [ ] **Niv (2009) — "Reinforcement learning in the brain"**
  Link: https://doi.org/10.1016/j.jmp.2008.12.005
  Section: Full review (Journal of Mathematical Psychology, 16 pages). Focus on Sections 2–4.
  Why: The best single review connecting RL algorithms to neuroscience. Covers model-free vs model-based learning, the role of dopamine, and how biological circuits implement actor-critic architectures. Excellent for interview preparation — demonstrates deep understanding of the field.
  Difficulty: Intermediate–Advanced | Time: 2 hours

---

## Three-Factor Learning Rules

- [ ] **Gerstner et al. (2018) — "Eligibility Traces and Plasticity on Behavioral Time Scales: Experimental Support of neoHebbian Three-Factor Learning Rules"**
  Link: https://doi.org/10.3389/fncir.2018.00053
  Section: Full paper (Frontiers in Neural Circuits, 12 pages). Focus on the three-factor framework (Section 2) and the experimental evidence (Section 3).
  Why: The definitive review of three-factor learning rules: Δw = M · e · f(pre, post), where M is the neuromodulatory signal (e.g. dopamine RPE), e is the eligibility trace, and f(pre, post) is the Hebbian/STDP term. This is precisely the learning rule NeuroDrive plans to implement. The paper also surveys experimental evidence supporting each component.
  Difficulty: Advanced | Time: 2–3 hours

- [ ] **Frémaux & Gerstner (2016) — "Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules"**
  Link: https://doi.org/10.3389/fncir.2015.00085
  Section: Full paper (Frontiers in Neural Circuits, 17 pages). Focus on Sections 1–3 (the theoretical framework) and Section 5 (connections to RL).
  Why: The theoretical companion to Gerstner et al. (2018). Provides the mathematical framework for three-factor rules and proves that they can solve RL problems. Section 5 explicitly connects three-factor rules to policy gradient methods — the bridge from NeuroDrive's current A2C to its future plasticity system.
  Difficulty: Advanced | Time: 3–4 hours

- [ ] **Kuśmierz, Isomura & Toyoizumi (2017) — "Learning with three factors: modulating Hebbian plasticity with errors"**
  Link: https://doi.org/10.1016/j.conb.2017.08.020
  Section: Full review (Current Opinion in Neurobiology, 7 pages).
  Why: A concise review focusing on the error-modulated interpretation of three-factor rules. Useful as a quick reference when implementing NeuroDrive's neuromodulatory gating — the "error" signal corresponds directly to the TD error δ_t computed by the critic.
  Difficulty: Advanced | Time: 1 hour
