# Linear Algebra for Machine Learning

## Prerequisites

- Comfortable with basic arithmetic (addition, multiplication, square roots).
- Familiarity with coordinate systems (x, y axes on a graph).
- No prior linear algebra knowledge required.

## Target Depth for This Project

**Level 2–3.** You need working fluency with vectors, dot products, and matrix-vector multiplication — enough to understand what a neural network layer computes, read weight matrices, and interpret norms. Proofs and abstract theory are not required.

---

## Core Concept

### Vectors: ordered lists of numbers

A **vector** is simply an ordered list of numbers. A 3D vector might be:

```
v = [3, 4, 0]
```

Each number is called a **component** or **element**. The number of components is the vector's **dimension** — here, 3.

**Geometric interpretation.** In two or three dimensions, a vector represents a point or an arrow in space. The vector [3, 4] points to the position 3 units along the x-axis and 4 units along the y-axis. In machine learning, vectors routinely live in hundreds or thousands of dimensions — impossible to visualise, but the algebra works identically.

### Magnitude (length) of a vector

The **magnitude** (or **L2 norm**) of a vector v measures how "long" the arrow is:

```
||v|| = √(v₁² + v₂² + ... + vₙ²)
```

**Worked example.** For v = [3, 4, 0]:

```
||v|| = √(3² + 4² + 0²)
     = √(9 + 16 + 0)
     = √25
     = 5
```

This generalises Pythagoras' theorem to any number of dimensions.

### Unit vectors and normalisation

A **unit vector** has magnitude exactly 1. To **normalise** any vector, divide each component by the magnitude:

```
v̂ = v / ||v|| = [3/5, 4/5, 0/5] = [0.6, 0.8, 0.0]
```

Normalisation preserves direction but standardises scale. It appears throughout ML whenever we care about direction rather than magnitude — for instance, cosine similarity between embeddings.

---

## Mathematical Foundation

### The dot product

The **dot product** (or inner product) of two vectors a and b of the same dimension is a single number:

**Algebraic definition:**

```
a · b = a₁b₁ + a₂b₂ + ... + aₙbₙ = Σ aᵢbᵢ
```

**Geometric definition:**

```
a · b = ||a|| · ||b|| · cos(θ)
```

where θ is the angle between the two vectors.

**Worked example (showing both agree).** Let a = [1, 2, 3] and b = [4, −1, 2].

Algebraic:

```
a · b = (1)(4) + (2)(−1) + (3)(2)
     = 4 − 2 + 6
     = 8
```

Geometric — first compute the magnitudes:

```
||a|| = √(1 + 4 + 9)   = √14 ≈ 3.742
||b|| = √(16 + 1 + 4)  = √21 ≈ 4.583
```

Using the algebraic result we can recover the angle:

```
cos(θ) = (a · b) / (||a|| · ||b||) = 8 / (3.742 × 4.583) ≈ 8 / 17.146 ≈ 0.4667
θ ≈ 62.2°
```

Plugging back in: ||a|| · ||b|| · cos(θ) = 3.742 × 4.583 × 0.4667 ≈ 8 ✓. Both definitions produce the same number.

**Key intuitions:**
- If the dot product is positive, the vectors broadly point in the same direction.
- If zero, they are perpendicular (orthogonal).
- If negative, they point in opposing directions.

### Matrix-vector multiplication

A **matrix** is a rectangular grid of numbers. A matrix W with m rows and n columns (written m × n) transforms an n-dimensional vector x into an m-dimensional vector y:

```
y = Wx
```

Each element of the output is the **dot product of one row of W with the input x**.

**Worked example.** Let W be a 2 × 3 matrix and x a 3D vector:

```
W = | 1   0   2 |      x = | 3 |
    | −1  3   1 |          | 1 |
                            | 2 |
```

Computing y = Wx:

```
y₁ = (1)(3) + (0)(1) + (2)(2) = 3 + 0 + 4 = 7
y₂ = (−1)(3) + (3)(1) + (1)(2) = −3 + 3 + 2 = 2
```

So y = [7, 2]. A 3D input has been mapped to a 2D output — the matrix changed the dimensionality.

### Affine transformations

An **affine transformation** combines matrix multiplication with a bias vector:

```
y = Wx + b
```

The matrix W can encode rotation, scaling, and shearing. The bias b adds translation (shifting). Together, they represent the most general "linear + shift" mapping, which is exactly what a single neural network layer computes.

---

## How NeuroDrive Uses This

**Linear layers.** Every dense layer in NeuroDrive's actor and critic networks computes y = Wx + b. The weight matrix W has shape (output_dim × input_dim). For a layer mapping 64 inputs to 32 outputs, W contains 64 × 32 = 2,048 learnable parameters, and the forward pass performs 32 dot products, each over 64 elements.

**L2 norm for weight health.** NeuroDrive's analytics system computes the L2 norm of each layer's parameter vector:

```
||θ|| = √(θ₁² + θ₂² + ... + θₙ²)
```

This single number summarises the overall magnitude of a layer's weights. Exploding norms signal instability; collapsing norms signal dead layers. NeuroDrive logs these per-layer norms every episode to monitor training health.

**Observation vectors.** The car's sensor readings — speed, track angle, distances to edges — are packed into a single vector. This vector is the network's input; the first layer multiplies it by a weight matrix, producing the first hidden representation.

---

## Common Misconceptions

1. **"Vectors must be 2D or 3D."** Vectors in ML are routinely 64-, 128-, or 1024-dimensional. The algebra does not change — only our ability to draw pictures.

2. **"Matrix multiplication is commutative."** It is not. Wx and xW are different operations (and xW may not even be defined). Order matters.

3. **"The dot product is a vector."** The dot product of two vectors is always a **scalar** (a single number), not another vector.

---

## Glossary

| Term | Definition |
|---|---|
| **Vector** | An ordered list of numbers; an element of ℝⁿ. |
| **Magnitude / L2 norm** | The length of a vector: ||v|| = √(Σ vᵢ²). |
| **Dot product** | A scalar computed as Σ aᵢbᵢ; measures alignment of two vectors. |
| **Matrix** | A rectangular grid of numbers with m rows and n columns. |
| **Affine transformation** | y = Wx + b; a linear map plus a translation. |
| **Unit vector** | A vector with magnitude 1. |
| **Normalisation** | Dividing a vector by its magnitude to produce a unit vector. |

---

## Recommended Materials

1. **3Blue1Brown — "Essence of Linear Algebra"** (YouTube series). Outstanding geometric intuition for vectors, matrices, and transformations — covers everything in this document visually.
2. **Mathematics for Machine Learning** — Deisenroth, Faisal & Ong (Cambridge University Press, free PDF). Chapters 2–3 cover vectors, matrices, and norms with ML context.
3. **Khan Academy — Linear Algebra** (khanacademy.org). Interactive exercises for dot products, matrix multiplication, and transformations — good for building procedural fluency.
