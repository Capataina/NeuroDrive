# Materials: Neural Networks

Resources for learning neural networks from first principles. NeuroDrive implements its entire neural network stack from scratch — no PyTorch, no TensorFlow — so deep conceptual understanding is essential.

---

## Foundations

- [ ] **3Blue1Brown — "Neural Networks" playlist (Episodes 1–3)**
  Link: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
  Section: Episode 1 "But what is a neural network?" (19 min), Episode 2 "Gradient descent, how neural networks learn" (21 min), Episode 3 "What is backpropagation really doing?" (14 min).
  Why: The best visual introduction to neural networks in existence. The animations make abstract concepts concrete — weighted sums, activations, cost functions, and the geometry of gradient descent. Watch these before reading anything else.
  Difficulty: Beginner | Time: 1 hour

- [ ] **Michael Nielsen — *Neural Networks and Deep Learning*, Chapters 1–2**
  Link: http://neuralnetworksanddeeplearning.com/
  Section: Chapter 1 "Using neural nets to recognize handwritten digits" and Chapter 2 "How the backpropagation algorithm works."
  Why: Nielsen builds a complete neural network for MNIST from scratch, explaining every design choice. The prose is unusually clear for a technical resource. Chapter 2's four fundamental equations of backpropagation are derived with care.
  Difficulty: Beginner–Intermediate | Time: 4–5 hours

- [ ] **Stanford CS231n — Lecture Notes: "Linear Classification" and "Neural Networks Part 1"**
  Link: https://cs231n.github.io/
  Section: "Linear classification" (loss functions, softmax) and "Neural Networks Part 1: Setting up the Architecture" (neurons, layers, activation functions).
  Why: Rigorous yet accessible notes that emphasise the computational graph perspective. The inline exercises and gradient computation examples are excellent practice.
  Difficulty: Intermediate | Time: 3 hours

- [ ] **Andrej Karpathy — "A Recipe for Training Neural Networks"**
  Link: http://karpathy.github.io/2019/04/25/recipe/
  Section: Full post.
  Why: Practical wisdom for debugging neural network training. Not theory — this is about the engineering mindset: overfit a single batch first, visualise everything, verify gradients numerically. Essential reading before implementing your MLP.
  Difficulty: Intermediate | Time: 30 minutes

---

## Backpropagation

- [ ] **3Blue1Brown — "Backpropagation calculus" (Episode 4)**
  Link: https://www.youtube.com/watch?v=tIeHLnjs5U8
  Section: Full video (10 min). Focuses on the chain rule applied to a simple 2-layer network.
  Why: The visual derivation of ∂C/∂w as a product of local derivatives along the computational graph is the clearest explanation you will find. Watch this before attempting the mathematical derivation.
  Difficulty: Beginner–Intermediate | Time: 15 minutes

- [ ] **Andrej Karpathy — "The spelled-out intro to neural networks and backpropagation: building micrograd"**
  Link: https://www.youtube.com/watch?v=VMj-3S1tku0
  Section: Full video (2:25:00). Key timestamps — 0:00–30:00 (derivatives and the chain rule), 30:00–1:15:00 (building the Value class with autograd), 1:15:00–1:50:00 (building a neuron/layer/MLP), 1:50:00–end (training loop and PyTorch comparison).
  Why: Karpathy builds a complete autograd engine from scratch in ~150 lines of Python, then uses it to train a neural network. This is the single best resource for understanding how backpropagation actually works in code. The approach mirrors NeuroDrive's cache-based backward pass.
  Difficulty: Intermediate | Time: 2.5 hours

- [ ] **Stanford CS231n — "Backpropagation, Intuitions"**
  Link: https://cs231n.github.io/optimization-2/
  Section: Full page. Focus on "Patterns in backward flow" (add gate, multiply gate, max gate) and "Staged computation."
  Why: The gate-level decomposition of backpropagation (each operation is a gate with a local gradient) is precisely how NeuroDrive implements its layer-by-layer backward pass. The "staged computation" examples show how to derive gradients for complex expressions step by step.
  Difficulty: Intermediate | Time: 1.5 hours

- [ ] **Michael Nielsen — Chapter 2: "How the backpropagation algorithm works"**
  Link: http://neuralnetworksanddeeplearning.com/chap2.html
  Section: Full chapter. Focus on the four fundamental equations (BP1–BP4) and the proof of each.
  Why: The most careful mathematical derivation of backpropagation available online. If you can reproduce all four equations on paper and explain the intuition behind each, you have mastered backpropagation.
  Difficulty: Intermediate–Advanced | Time: 2–3 hours

---

## Optimisers

- [ ] **Kingma & Ba (2015) — "Adam: A Method for Stochastic Optimization"**
  Link: https://arxiv.org/abs/1412.6980
  Section: Sections 1–2 (Algorithm 1 in particular). The paper is short (9 pages) and the algorithm description is self-contained.
  Why: NeuroDrive implements Adam from scratch. You must understand the first and second moment estimates (m and v), bias correction, and the role of ε. Compare Algorithm 1 directly to NeuroDrive's `optim.rs`.
  Difficulty: Intermediate–Advanced | Time: 1.5 hours

- [ ] **Sebastian Ruder (2016) — "An overview of gradient descent optimization algorithms"**
  Link: https://ruder.io/optimizing-gradient-descent/
  Section: Full post. Focus on sections covering SGD, Momentum, RMSProp, and Adam.
  Why: The best single resource comparing optimisers. The progression SGD → Momentum → RMSProp → Adam shows how each improvement addresses a specific failure mode. The animated GIFs of optimisers navigating loss surfaces are particularly instructive.
  Difficulty: Intermediate | Time: 1 hour

- [ ] **Stanford CS231n — "Neural Networks Part 3: Learning and Evaluation"**
  Link: https://cs231n.github.io/neural-networks-3/
  Section: "Parameter updates" section (SGD, Momentum, Adam) and "Hyperparameter optimization" section.
  Why: Practical guidance on learning rate schedules, batch sizes, and weight decay. The advice on gradient checking (comparing analytic gradients to numerical gradients) is essential for debugging your from-scratch implementation.
  Difficulty: Intermediate | Time: 1.5 hours

---

## From-Scratch Implementations

- [ ] **Andrej Karpathy — "makemore" series (Parts 1–5)**
  Link: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
  Section: Part 1 "bigram model" (1:57:00), Part 2 "MLP" (1:15:00), Part 3 "Activations/Gradients/BatchNorm" (1:55:00). Parts 4–5 cover RNNs and transformers (optional for NeuroDrive).
  Why: The entire series builds language models from scratch with manual gradient computation. Part 3 is particularly relevant — Karpathy diagnoses dead neurons, exploding gradients, and poor initialisations using histograms. This debugging mindset is essential for NeuroDrive's handwritten MLP.
  Difficulty: Intermediate–Advanced | Time: 5+ hours (across parts)

- [ ] **fast.ai — "From the Foundations" (Part 2 of Practical Deep Learning)**
  Link: https://course.fast.ai/Lessons/part2.html
  Section: Lessons 1–5 (building a training framework from scratch). Focus on Lesson 2 (forward and backward pass) and Lesson 3 (optimisers).
  Why: Jeremy Howard rebuilds PyTorch's core abstractions from scratch. Watching someone construct Module, Parameter, and Optimizer classes from nothing builds intuition for NeuroDrive's similar Rust abstractions.
  Difficulty: Advanced | Time: 5+ hours (across lessons)

- [ ] **George Hotz — "tinygrad" walkthrough**
  Link: https://github.com/tinygrad/tinygrad
  Section: Read `tinygrad/tensor.py` and `tinygrad/mlops.py` (~500 lines total). Focus on the lazy evaluation and backward pass.
  Why: A minimal autograd engine in Python that is small enough to read end-to-end. Comparing tinygrad's approach (lazy ops + automatic differentiation) to NeuroDrive's approach (eager forward pass + manual layer-by-layer backward) illuminates the trade-offs of each design.
  Difficulty: Advanced | Time: 2–3 hours
