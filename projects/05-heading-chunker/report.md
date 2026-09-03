# Lesson 05, top-k chunks, fixed vs headings

Question: **What is overfitting in a neural network?**  
Keywords required (relevance=`all`, all must be in one chunk): `Overfitting` · `Training accuracy` · `Validation accuracy`  
Config: `bge-small` · chunk-size 1000 · top-5 · overlap 50

Open this file in the Markdown **preview**, each retrieved chunk is a collapsible block so you can scan ten of them without scrolling forever.

---

## Rank summary

| Chunker | Chunks in index | Top-5 verdict | Rank in top-5 | Rank across *all* chunks (if not in top-5) | Coverage |
|---|---|---|---|---|---|
| `fixed` | 734 | ❌ MISS |, | #8 | 33% |
| `headings` | 1499 | ❌ MISS |, | #67 | 33% |

---

## `fixed` chunker, top-5 retrieved

<details open>
<summary><b>#1</b> · sim 0.751 · <code>day23-oss-finetuning.md</code> · 998 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
s during training

**Purpose:**
- Prevents overfitting
- Forces model to learn robust features
- Can't rely on specific neurons

**How it works:**
- During training: Randomly zero out 10-20% of activations
- During inference: Use all neurons (no dropout)

**Setting:** 0.1 (10%) is standard, 0.2 (20%) for more regularization

### Weight Decay

**Definition:** Penalty for large weights (L2 regularization)

**Purpose:**
- Prevents overfitting
- Encourages simpler models
- Typical value: 0.01

### Max Sequence Length

**Definition:** Maximum number of tokens in input

**Why it matters:**
- Memory usage scales with sequence length
- Longer sequences = more memory = smaller batch size

**Strategy:**
- Analyze your data: What's the 95th percentile length?
- Cut off at that point
- Example: If 95% of data is <110 tokens, use max_length=110
- Trade-off: Lose some information vs. train more efficiently

---

## 8. Data Preparation

Proper data preparation is crucial for successful fine-tuning.
```
</details>

<details open>
<summary><b>#2</b> · sim 0.717 · <code>day26-deep-learning.md</code> · 999 chars · Overfitting ✗ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
* (penalize large weights)
4. **Data augmentation** (rotate, flip, zoom images)
5. **Early stopping** (stop when validation worsens)
6. **Reduce model complexity** (fewer layers/neurons)

---

## Module 7: Key Takeaways

### The Big Picture

**Deep Learning = Forward Propagation + Backward Propagation**

**Forward Propagation:**
1. Input data
2. Apply weights
3. Add bias
4. Apply activation function
5. Move to next layer
6. Generate prediction

**Backward Propagation:**
1. Calculate loss (error)
2. Compute gradients
3. Update weights using optimizer
4. Repeat

**Training = Repeating this cycle many times (epochs)**

---

### Essential Concepts Summary

**1. Neural Networks**
- Inspired by human brain
- Learn through examples
- Consist of layers of neurons

**2. Weights and Bias**
- Weights: Importance of each input
- Bias: Shift in decision boundary
- Both learned during training

**3. Activation Functions**
- Introduce non-linearity
- Enable complex pattern learning
- Choose based on
```
</details>

<details open>
<summary><b>#3</b> · sim 0.709 · <code>day27-deep-learning-2.md</code> · 998 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Classification Head
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Architecture Pattern: Progressive Deepening**

```
Block 1 (32 filters) → Block 2 (64 filters) → Block 3 (128 filters) → Dense Head
```

Each block doubles the number of filters while halving spatial dimensions. Dropout increases with depth (0.2 → 0.3 → 0.4 → 0.5) because deeper layers are more prone to overfitting.

#### Advanced Training for CIFAR-10

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Data augmentation (more aggressive for CIFAR-10)
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
```
</details>

<details open>
<summary><b>#4</b> · sim 0.706 · <code>day27-deep-learning-2.md</code> · 994 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
ation Problem

Before ResNet, researchers observed a paradox: deeper networks performed *worse* than shallow ones, even on training data. This wasn't overfitting, it was a fundamental training problem.

**The problem:**
```
20-layer network: 92% accuracy
56-layer network: 90% accuracy   ← WORSE, even on training data!
```

Deeper networks should be at least as good as shallow ones (the extra layers could just learn identity functions), but gradient-based training couldn't achieve this.

#### The ResNet Solution: Skip Connections

**Key Insight:** Instead of learning a mapping H(x), learn the *residual* F(x) = H(x) - x. Then the output is F(x) + x.

```
Traditional block:           Residual block:
    Input x                     Input x ─────────────┐
       ↓                           ↓                  │
   Conv + ReLU                 Conv + ReLU            │
       ↓                           ↓                  │
   Conv + ReLU                 Conv + ReLU            │
       ↓
```
</details>

<details open>
<summary><b>#5</b> · sim 0.701 · <code>day26-deep-learning.md</code> · 999 chars · Overfitting ✗ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
rward and backward propagation.**
A: Forward propagation passes input through layers to generate predictions. Backward propagation calculates gradients and updates weights to minimize loss.

**Q: What activation function should I use?**
A: ReLU for hidden layers (fast, effective), Sigmoid for binary classification output, Softmax for multi-class output.

**Q: Why do we need activation functions?**
A: Without them, the network would be just a linear combination, unable to learn complex non-linear patterns.

**Q: What is the role of bias?**
A: Bias allows the model to shift the decision boundary, improving model flexibility and fit.

**Q: What optimizer should I use?**
A: Adam is the industry standard - it's robust, converges fast, and works well for most problems.

**Q: How do CNNs differ from ANNs?**
A: CNNs preserve spatial structure, use weight sharing through filters, and are designed for image/spatial data. ANNs flatten inputs and are general-purpose.

**Q: What is pooling and why
```
</details>

> The answer chunk is **not** in the top-5 above. It exists in the index at full-corpus rank **#8** of 734, in the net, but out-ranked by 7 other chunks.

---

## `headings` chunker, top-5 retrieved

<details open>
<summary><b>#1</b> · sim 0.802 · <code>day26-deep-learning.md</code> · 360 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
#### **Preventing Overfitting**

**Techniques:**
1. **More training data** (best solution)
2. **Dropout layers** (randomly deactivate neurons)
3. **L1/L2 regularization** (penalize large weights)
4. **Data augmentation** (rotate, flip, zoom images)
5. **Early stopping** (stop when validation worsens)
6. **Reduce model complexity** (fewer layers/neurons)

---
```
</details>

<details open>
<summary><b>#2</b> · sim 0.727 · <code>day27-deep-learning-2.md</code> · 927 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
ivation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.4),

    # Classification Head
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

**Architecture Pattern: Progressive Deepening**

```
Block 1 (32 filters) → Block 2 (64 filters) → Block 3 (128 filters) → Dense Head
```

Each block doubles the number of filters while halving spatial dimensions. Dropout increases with depth (0.2 → 0.3 → 0.4 → 0.5) because deeper layers are more prone to overfitting.
```
</details>

<details open>
<summary><b>#3</b> · sim 0.723 · <code>day24-finetuning-showdown.md</code> · 229 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
l attention: `["q_proj", "k_proj", "v_proj", "o_proj"]`
  - Attention + FFN: `[..., "gate_proj", "up_proj", "down_proj"]`

**4. Dropout:**
- Regularization to prevent overfitting
- Typical: 0.05 or 0.1
- Higher for small datasets
```
</details>

<details open>
<summary><b>#4</b> · sim 0.719 · <code>day26-deep-learning.md</code> · 905 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
combination, unable to learn complex non-linear patterns.

**Q: What is the role of bias?**
A: Bias allows the model to shift the decision boundary, improving model flexibility and fit.

**Q: What optimizer should I use?**
A: Adam is the industry standard - it's robust, converges fast, and works well for most problems.

**Q: How do CNNs differ from ANNs?**
A: CNNs preserve spatial structure, use weight sharing through filters, and are designed for image/spatial data. ANNs flatten inputs and are general-purpose.

**Q: What is pooling and why use it?**
A: Pooling reduces spatial dimensions while keeping important features. It reduces computation, adds translation invariance, and prevents overfitting.

**Q: How many epochs should I train for?**
A: Start with 10-20 and monitor validation loss. Use early stopping to prevent overfitting. The exact number depends on dataset size and complexity.

---
```
</details>

<details open>
<summary><b>#5</b> · sim 0.715 · <code>day23-oss-finetuning.md</code> · 231 chars · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

```text
### Pitfall 3: Overfitting

**Symptoms:** Training loss decreases, validation loss increases

**Solutions:**
- Reduce number of epochs
- Increase dropout
- Add more training data
- Use weight decay
- Reduce model capacity (lower r)
```
</details>

> The answer chunk is **not** in the top-5 above. It exists in the index at full-corpus rank **#67** of 1499, in the net, but out-ranked by 66 other chunks.

---
