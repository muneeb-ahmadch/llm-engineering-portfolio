# Fine-Tuning Showdown: Lecture Notes

## Introduction: The Fine-Tuning Decision

Welcome to the capstone project that brings together everything you've learned about LLM fine-tuning. This isn't just about learning two techniques—it's about becoming a practitioner who can make informed decisions about WHICH technique to use and WHY.

### The Real-World Problem

Imagine you're a machine learning engineer at a healthcare startup. Your CEO asks:

> "We need to build a medical Q&A assistant. Should we use OpenAI's fine-tuning or train our own model with QLoRA?"

This is the question thousands of companies face every day. There's no universal "right" answer—it depends on your constraints, priorities, and use case.

---

## Lecture 1: Understanding the Landscape

### The Two Paths

**Path 1: Proprietary Fine-Tuning (OpenAI)**
- You send your data to OpenAI
- They fine-tune their model on their infrastructure
- You get back a model ID
- You call it via API
- They handle everything: servers, scaling, updates

**Path 2: Open-Source Fine-Tuning (QLoRA)**
- You download an open-source model (Llama, Mistral, etc.)
- You fine-tune it on your own GPU
- You own the weights
- You deploy it however you want
- You manage everything

### Why This Matters

**Real Example 1: Startup with $10K Budget**
- Needs to process 1M medical queries/month
- OpenAI cost: ~$1,500/month ongoing
- QLoRA cost: $2,000 one-time (GPU) + $200/month (hosting)
- **Decision:** QLoRA breaks even in 2 months

**Real Example 2: Hospital with HIPAA Requirements**
- Cannot send patient data to third parties
- Must run models on-premise
- **Decision:** QLoRA is the only option

**Real Example 3: Rapid Prototype for Demo**
- Need working demo in 2 days
- No ML infrastructure
- **Decision:** OpenAI (deploy in hours)

### The Comparison Framework

We'll evaluate both approaches across 5 dimensions:

1. **Performance** - Accuracy, quality, consistency
2. **Cost** - Training + inference + infrastructure
3. **Speed** - Time to deploy + inference latency
4. **Control** - Customization, privacy, ownership
5. **Complexity** - Engineering effort, maintenance

---

## Lecture 2: The Medical Q&A Use Case

### Why Medical Questions?

We chose medical Q&A for several strategic reasons:

**1. Clear Evaluation Metric**
- Multiple choice questions have objective correct answers
- Easy to calculate accuracy: correct / total
- No subjective judgment needed

**2. High-Value Domain**
- Medical knowledge is specialized
- General LLMs perform poorly (60-70% accuracy)
- Fine-tuning shows clear improvement (80-90% accuracy)
- Real business value: saves doctors time, helps students

**3. Safety-Critical Application**
- Wrong medical advice can harm people
- Teaches responsible AI development
- Forces you to think about model reliability

**4. Publicly Available Data**
- MedQA dataset is free and high-quality
- Based on real medical board exams (USMLE)
- No licensing issues

### The MedQA Dataset

**Source:** Medical licensing exam questions (USMLE-style)

**Format:**
```
Question: A 65-year-old man with a history of hypertension...
Options:
  A) Myocardial infarction
  B) Pulmonary embolism
  C) Aortic dissection
  D) Pneumothorax
Correct Answer: A
Explanation: The patient's symptoms (chest pain radiating to jaw, 
elevated troponin) are classic for MI...
```

**Topics Covered:**
- Anatomy & Physiology
- Pharmacology
- Pathology
- Internal Medicine
- Surgery
- Pediatrics
- Psychiatry

**Dataset Statistics:**
- Total: ~12,000 questions
- Average question length: 100-200 words
- Average explanation length: 50-100 words
- Difficulty: Medical school / board exam level

### Baseline Performance

**Without Fine-Tuning:**
- GPT-3.5: ~65% accuracy (random is 25%)
- GPT-4: ~75% accuracy
- Llama 3.2 (7B): ~60% accuracy

**Why So Low?**
- Medical terminology is specialized
- Requires understanding of physiology, not just facts
- Context matters (patient history, symptoms)
- Distractors are designed to be plausible

**Expected After Fine-Tuning:**
- OpenAI fine-tuned: 80-85% accuracy
- QLoRA fine-tuned: 78-83% accuracy

The improvement of 15-20% is significant and demonstrates the value of domain-specific fine-tuning.

---

## Lecture 3: OpenAI Fine-Tuning Deep Dive

### How OpenAI Fine-Tuning Works

**Architecture (Simplified):**
```
Your Training Data (JSONL)
        ↓
OpenAI API Upload
        ↓
Proprietary Training Infrastructure
(You don't see this part)
        ↓
Fine-Tuned Model ID
(e.g., ft:gpt-4o-mini-2024-07-18:org:suffix:id)
        ↓
API Endpoint for Inference
```

### Data Format Requirements

OpenAI requires a specific JSONL format:

**❌ Wrong Format:**
```json
{"question": "What is the treatment?", "answer": "Antibiotics"}
```

**✅ Correct Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a medical expert."},
    {"role": "user", "content": "Question: What is the treatment for...?"},
    {"role": "assistant", "content": "The correct answer is A. Antibiotics..."}
  ]
}
```

**Why This Format?**
- Matches the chat completion API structure
- Allows system prompts for behavior control
- Supports multi-turn conversations
- Consistent with how the model was pre-trained

### Key Hyperparameters

| Parameter | Recommended Value | Explanation |
|-----------|------------------|-------------|
| `model` | `gpt-4o-mini-2024-07-18` | Cost-effective, good performance |
| `n_epochs` | 3-4 | More epochs = more training passes |
| `batch_size` | Auto | Let OpenAI optimize |
| `learning_rate_multiplier` | 1.0 | Default works well |
| `suffix` | "medical-qa" | Helps identify your model |

**Epochs Explained:**
- 1 epoch = model sees each example once
- Too few (1-2): Underfitting, poor performance
- Just right (3-4): Good balance
- Too many (5+): Overfitting, memorization

**How to Choose Epochs:**
- Small dataset (<1K examples): 4-5 epochs
- Medium dataset (1K-10K): 3-4 epochs
- Large dataset (>10K): 2-3 epochs

### Cost Breakdown

**Training Costs (as of 2026):**
- Input tokens: $0.03 per 1K tokens
- Output tokens: $0.06 per 1K tokens

**Example Calculation:**
```
Dataset: 2,000 medical Q&A examples
Average input: 150 tokens (question + options)
Average output: 50 tokens (answer + explanation)
Epochs: 3

Training tokens:
  Input: 2,000 × 150 × 3 = 900,000 tokens
  Output: 2,000 × 50 × 3 = 300,000 tokens

Training cost:
  Input: 900K × $0.03/1K = $27
  Output: 300K × $0.06/1K = $18
  Total: $45
```

**Inference Costs:**
- Input: $0.0015 per 1K tokens
- Output: $0.002 per 1K tokens

**Example:**
```
1,000 inferences
Average input: 150 tokens
Average output: 50 tokens

Cost: (150 × $0.0015 + 50 × $0.002) × 1 = $0.325
Per 1K inferences: ~$0.33
```

### Monitoring Training

OpenAI provides training metrics:

**What to Watch:**
1. **Training Loss** - Should decrease over epochs
2. **Validation Loss** - Should decrease but not diverge from training
3. **Training Tokens** - Total tokens processed
4. **Status** - Running, succeeded, failed

**Good Training:**
```
Epoch 1: train_loss=1.2, val_loss=1.3
Epoch 2: train_loss=0.8, val_loss=0.9
Epoch 3: train_loss=0.6, val_loss=0.7
```

**Overfitting Warning:**
```
Epoch 1: train_loss=1.2, val_loss=1.3
Epoch 2: train_loss=0.8, val_loss=0.9
Epoch 3: train_loss=0.4, val_loss=1.1  ← Val loss increased!
```

### Advantages of OpenAI Fine-Tuning

**1. Speed to Deployment**
- Upload data → Start training → Get model ID
- Entire process: 30 minutes to 2 hours
- No infrastructure setup needed

**2. Managed Infrastructure**
- OpenAI handles servers, GPUs, scaling
- Automatic updates and improvements
- 99.9% uptime SLA

**3. Low Engineering Overhead**
- No model management
- No deployment complexity
- No GPU maintenance

**4. Proven Reliability**
- Battle-tested infrastructure
- Used by thousands of companies
- Enterprise support available

**5. Fast Inference**
- Optimized serving infrastructure
- ~200-500ms latency
- Auto-scaling for traffic spikes

### Disadvantages of OpenAI Fine-Tuning

**1. Recurring Costs**
- Pay per inference forever
- Costs scale linearly with usage
- Can become expensive at high volume

**2. Data Privacy Concerns**
- Training data sent to OpenAI
- Subject to their data policies
- Not suitable for highly sensitive data

**3. Limited Customization**
- Can't modify model architecture
- Limited control over training process
- Can't inspect model internals

**4. Vendor Lock-In**
- Dependent on OpenAI's pricing
- Dependent on OpenAI's availability
- Hard to migrate away

**5. No Offline Capability**
- Requires internet connection
- Can't run in air-gapped environments
- Latency includes network round-trip

### When to Use OpenAI Fine-Tuning

**✅ Good Fit:**
- Rapid prototyping and MVPs
- Low to medium inference volume (<100K/day)
- Budget for recurring API costs
- Need fast deployment (days not weeks)
- Don't have ML infrastructure
- Data privacy is not critical

**❌ Poor Fit:**
- High inference volume (>1M/day)
- Strict data privacy requirements
- Need offline/edge deployment
- Want full model control
- Long-term cost optimization critical

---

## Lecture 4: QLoRA Fine-Tuning Deep Dive

### The QLoRA Innovation

**The Problem:**
- Llama 3.2 (7B parameters) = ~13GB in memory
- Full fine-tuning requires 3-4x memory (gradients, optimizer states)
- Total: ~40-50GB GPU memory needed
- Most people don't have 80GB A100 GPUs

**The QLoRA Solution:**
```
Step 1: 4-bit Quantization
  13GB model → 3.2GB (75% reduction!)

Step 2: LoRA Adapters
  Train only small adapter matrices (~0.6% of parameters)
  Adapters: ~100MB

Step 3: Efficient Training
  Use paged optimizers (bitsandbytes)
  Total memory: ~4-6GB

Result: Fine-tune 7B model on 24GB consumer GPU!
```

### Understanding 4-bit Quantization

**Normal Precision (FP16):**
- Each weight = 16 bits = 2 bytes
- 7B parameters × 2 bytes = 14GB

**4-bit Quantization (NF4):**
- Each weight = 4 bits = 0.5 bytes
- 7B parameters × 0.5 bytes = 3.5GB

**How Does It Work?**
1. **Quantize weights** to 4-bit during loading
2. **Dequantize to 16-bit** during computation
3. **Compute in higher precision** for accuracy
4. **Results stay accurate** (minimal quality loss)

**NF4 (Normal Float 4-bit):**
- Specially designed for neural networks
- Preserves important weight distributions
- Better than naive 4-bit quantization

### Understanding LoRA

**Full Fine-Tuning:**
```
Update ALL 7 billion parameters
Requires huge memory for gradients
Slow and expensive
```

**LoRA (Low-Rank Adaptation):**
```
Freeze original model weights
Add small trainable matrices (adapters)
Only train the adapters (~43M parameters)
Merge adapters at inference time
```

**Mathematical Intuition:**
```
Original weight matrix: W (large, frozen)
LoRA decomposition: W + A × B

Where:
  A = matrix of size (d × r)
  B = matrix of size (r × d)
  r = rank (typically 8-64)

Example:
  W = 4096 × 4096 = 16M parameters (frozen)
  A = 4096 × 16 = 65K parameters (trainable)
  B = 16 × 4096 = 65K parameters (trainable)
  Total trainable: 130K (vs 16M!)
```

### LoRA Configuration Explained

```python
lora_config = LoraConfig(
    r=16,              # Rank of adapter matrices
    lora_alpha=32,     # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Parameter Guide:**

**1. Rank (r):**
- Controls adapter capacity
- Higher r = more parameters = more expressive
- Typical values: 8, 16, 32, 64
- Trade-off: capacity vs speed vs memory

| Rank | Trainable % | Use Case |
|------|-------------|----------|
| 4 | ~0.3% | Simple tasks, very limited data |
| 8 | ~0.6% | Most tasks, good default |
| 16 | ~1.2% | Complex tasks, more data |
| 32 | ~2.4% | Very complex tasks |

**2. Alpha:**
- Scaling factor for adapter outputs
- Typically set to 2× rank
- r=8 → alpha=16
- r=16 → alpha=32

**3. Target Modules:**
- Which layers get adapters
- Common choices:
  - Attention only: `["q_proj", "v_proj"]`
  - All attention: `["q_proj", "k_proj", "v_proj", "o_proj"]`
  - Attention + FFN: `[..., "gate_proj", "up_proj", "down_proj"]`

**4. Dropout:**
- Regularization to prevent overfitting
- Typical: 0.05 or 0.1
- Higher for small datasets

### Training Configuration

```python
training_args = TrainingArguments(
    output_dir="./medical-qa-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit"
)
```

**Key Parameters:**

**Effective Batch Size:**
```
Effective batch = per_device_batch × gradient_accumulation × num_gpus
Example: 4 × 4 × 1 = 16
```

**Learning Rate:**
- QLoRA typically uses higher LR than full fine-tuning
- Typical range: 1e-4 to 5e-4
- Start with 2e-4

**Scheduler:**
- `cosine`: Smooth decay (recommended)
- `linear`: Linear decay
- `constant`: No decay

**Optimizer:**
- `paged_adamw_8bit`: Memory-efficient Adam
- Reduces memory by ~50%
- Essential for QLoRA

### Cost and Resource Analysis

**GPU Requirements:**

| Model Size | Quantized Size | Training Memory | Recommended GPU |
|------------|----------------|-----------------|-----------------|
| Llama 3.2-1B | 0.8GB | 2-3GB | T4 (16GB) |
| Llama 3.2-3B | 2.2GB | 4-6GB | T4 (16GB) |
| Llama 3.2-7B | 3.5GB | 6-8GB | L4 (24GB) |

**Training Time:**

| Dataset Size | Model | GPU | Time |
|--------------|-------|-----|------|
| 2K examples | 3B | T4 | 20-30 min |
| 8K examples | 3B | T4 | 1-2 hours |
| 8K examples | 7B | L4 | 1-2 hours |

**Training Cost (Cloud GPU):**

| Provider | GPU | Cost/Hour | 2hr Training |
|----------|-----|-----------|--------------|
| Google Colab | T4 | Free | $0 |
| Colab Pro | T4/A100 | $10/month | $0.33 |
| Lambda Labs | A10 | $0.60 | $1.20 |
| RunPod | RTX 4090 | $0.40 | $0.80 |

**Inference Cost:**
- Self-hosted: $0 (if you own GPU)
- Cloud hosting: ~$0.02-0.05 per 1K inferences
- Much cheaper than OpenAI at scale

### Advantages of QLoRA

**1. Cost-Effective at Scale**
- One-time training cost
- Zero or low inference cost
- Break-even quickly with high volume

**2. Full Control**
- Own the model weights
- Modify architecture if needed
- Control deployment environment

**3. Data Privacy**
- Train on-premise
- Data never leaves your infrastructure
- Meets HIPAA, GDPR requirements

**4. Offline Capability**
- Run without internet
- Deploy on edge devices
- Air-gapped environments

**5. Customization**
- Experiment with architectures
- Try different base models
- Combine multiple adapters

### Disadvantages of QLoRA

**1. Higher Engineering Complexity**
- Need to manage infrastructure
- Handle model deployment
- Debug training issues

**2. Slower Inference**
- 500ms - 2000ms per inference
- Depends on hardware
- 3-5x slower than OpenAI

**3. Longer Time to Deploy**
- Setup: days to weeks
- Training: hours
- Deployment: configuration needed

**4. Maintenance Burden**
- Monitor model performance
- Handle updates
- Manage GPU resources

**5. Requires ML Expertise**
- Understand hyperparameters
- Debug training issues
- Optimize inference

### When to Use QLoRA

**✅ Good Fit:**
- High inference volume (>100K/day)
- Data privacy is critical
- Budget for GPU infrastructure
- Have ML engineering team
- Long-term deployment (>6 months)
- Need offline capability

**❌ Poor Fit:**
- Rapid prototyping (use OpenAI first)
- Low inference volume
- No ML expertise
- Can't invest in infrastructure
- Short-term project

---

## Lecture 5: Building the Evaluation Framework

### Why Evaluation Matters

You can't improve what you don't measure. A rigorous evaluation framework lets you:

1. **Compare fairly** - Same test set, same metrics
2. **Identify weaknesses** - Where does each model fail?
3. **Make decisions** - Which approach is better for your use case?
4. **Communicate results** - Show stakeholders clear data

### Core Metrics for Medical Q&A

**1. Accuracy**
```python
accuracy = correct_answers / total_questions
```

**What it measures:** Overall correctness

**Example:**
- 850 correct out of 1,000 questions
- Accuracy = 85%

**2. Precision (per option)**
```python
precision = true_positives / (true_positives + false_positives)
```

**What it measures:** When model predicts A, how often is it correct?

**3. Recall (per option)**
```python
recall = true_positives / (true_positives + false_negatives)
```

**What it measures:** Of all correct A answers, how many did model find?

**4. F1 Score**
```python
f1 = 2 × (precision × recall) / (precision + recall)
```

**What it measures:** Harmonic mean of precision and recall

### Performance Evaluation Code

```python
def evaluate_model(model, test_dataset):
    correct = 0
    total = 0
    predictions = []
    
    for example in test_dataset:
        # Get model prediction
        pred = model.predict(example['question'])
        
        # Check if correct
        is_correct = (pred == example['answer'])
        correct += is_correct
        total += 1
        
        predictions.append({
            'question_id': example['id'],
            'predicted': pred,
            'actual': example['answer'],
            'correct': is_correct
        })
    
    return {
        'accuracy': correct / total,
        'total': total,
        'predictions': predictions
    }
```

### Latency Measurement

**Why Latency Matters:**
- User experience (users wait for responses)
- System capacity (requests per second)
- Cost (longer inference = more compute)

**What to Measure:**
```python
import time

latencies = []
for example in test_set:
    start = time.time()
    response = model.predict(example['question'])
    latency = time.time() - start
    latencies.append(latency)

# Calculate statistics
avg_latency = sum(latencies) / len(latencies)
p50_latency = np.percentile(latencies, 50)  # Median
p95_latency = np.percentile(latencies, 95)  # 95th percentile
p99_latency = np.percentile(latencies, 99)  # 99th percentile
```

**Why P95/P99?**
- Average can hide outliers
- P95 = 95% of requests are faster than this
- P99 = 99% of requests are faster than this
- Important for user experience guarantees

### Cost Analysis Framework

**OpenAI Cost Calculation:**
```python
# Training cost
training_tokens = num_examples * avg_tokens_per_example * epochs
training_cost = training_tokens * cost_per_1k_tokens / 1000

# Inference cost (per 1K requests)
avg_input_tokens = 150
avg_output_tokens = 50
inference_cost_per_1k = (
    (avg_input_tokens * input_cost_per_1k / 1000) +
    (avg_output_tokens * output_cost_per_1k / 1000)
) * 1000
```

**QLoRA Cost Calculation:**
```python
# Training cost (cloud GPU)
training_hours = 2  # Measured
gpu_cost_per_hour = 0.60  # Lambda Labs A10
training_cost = training_hours * gpu_cost_per_hour

# Inference cost (self-hosted)
# If you own the GPU: $0
# If cloud-hosted: similar calculation to OpenAI but cheaper
```

**Break-Even Analysis:**
```python
def calculate_breakeven(openai_training, openai_per_1k,
                       qlora_training, qlora_per_1k):
    """
    Calculate how many inferences until QLoRA is cheaper
    """
    cost_difference = qlora_training - openai_training
    per_1k_savings = openai_per_1k - qlora_per_1k
    
    if per_1k_savings <= 0:
        return float('inf')  # QLoRA never cheaper
    
    breakeven_1k_requests = cost_difference / per_1k_savings
    breakeven_total_requests = breakeven_1k_requests * 1000
    
    return breakeven_total_requests

# Example
breakeven = calculate_breakeven(
    openai_training=20,
    openai_per_1k=0.15,
    qlora_training=2,
    qlora_per_1k=0.02
)
print(f"Break-even at {breakeven:,.0f} requests")
# Output: Break-even at 138,462 requests
```

### Quality Evaluation

Beyond accuracy, evaluate response quality:

**1. Explanation Quality**
- Does the model provide reasoning?
- Is the explanation medically accurate?
- Is it understandable?

**2. Confidence Calibration**
- When model is confident, is it usually right?
- When uncertain, does it express uncertainty?

**3. Error Analysis**
```python
# Categorize errors
errors = {
    'misread_question': 0,
    'wrong_medical_knowledge': 0,
    'confused_options': 0,
    'other': 0
}

for pred in predictions:
    if not pred['correct']:
        # Manually categorize or use LLM to categorize
        category = categorize_error(pred)
        errors[category] += 1
```

### Comparison Visualization

**Create comparison charts:**

```python
import matplotlib.pyplot as plt

# Accuracy comparison
models = ['OpenAI', 'QLoRA']
accuracies = [0.853, 0.838]

plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0.8, 0.9)
plt.show()

# Latency comparison
latencies = [215, 850]  # milliseconds

plt.bar(models, latencies)
plt.ylabel('Latency (ms)')
plt.title('Average Inference Latency')
plt.show()

# Cost comparison (at different volumes)
volumes = [1000, 10000, 100000, 1000000]
openai_costs = [20 + v * 0.15 / 1000 for v in volumes]
qlora_costs = [2 + v * 0.02 / 1000 for v in volumes]

plt.plot(volumes, openai_costs, label='OpenAI')
plt.plot(volumes, qlora_costs, label='QLoRA')
plt.xlabel('Number of Inferences')
plt.ylabel('Total Cost ($)')
plt.title('Cost Comparison')
plt.legend()
plt.xscale('log')
plt.show()
```

---

## Lecture 6: Creating the Comparison Report

### Report Structure

A good comparison report tells a story:

1. **What did you do?** (Methodology)
2. **What did you find?** (Results)
3. **What does it mean?** (Analysis)
4. **What should we do?** (Recommendations)

### Section 1: Executive Summary

**Purpose:** Busy executives read only this

**What to include:**
- One-sentence project description
- Key finding (which approach is better?)
- Main recommendation
- Critical numbers (accuracy, cost, latency)

**Example:**
```
We fine-tuned GPT-4o-mini (OpenAI) and Llama 3.2 (QLoRA) on 8,000 
medical Q&A examples and evaluated on 1,000 test questions. OpenAI 
achieved 85.3% accuracy with 215ms latency at $0.15 per 1K inferences, 
while QLoRA achieved 83.8% accuracy with 850ms latency at $0.02 per 1K 
inferences. For high-volume production (>100K inferences/day), we 
recommend QLoRA due to 87% cost savings despite 1.5% lower accuracy 
and 4x higher latency.
```

### Section 2: Methodology

**Purpose:** Reproducibility and credibility

**What to include:**
- Dataset description and split
- Training configuration for both models
- Evaluation metrics and test set
- Hardware and software versions

**Example Table:**
| Aspect | OpenAI | QLoRA |
|--------|--------|-------|
| Model | GPT-4o-mini | Llama 3.2-3B |
| Training Examples | 2,000 | 8,000 |
| Epochs | 3 | 3 |
| Batch Size | Auto | 4 (effective: 16) |
| Learning Rate | Auto | 2e-4 |
| Training Time | 25 min | 90 min |
| Hardware | N/A | Lambda Labs A10 |

### Section 3: Performance Results

**Present results clearly:**

**Accuracy Comparison:**
```
| Metric | OpenAI | QLoRA | Difference |
|--------|--------|-------|------------|
| Accuracy | 85.3% | 83.8% | -1.5% |
| Precision | 85.7% | 84.1% | -1.6% |
| Recall | 85.3% | 83.8% | -1.5% |
| F1 Score | 85.5% | 83.9% | -1.6% |
```

**Latency Distribution:**
```
| Percentile | OpenAI | QLoRA |
|------------|--------|-------|
| P50 (median) | 205ms | 780ms |
| P95 | 280ms | 1200ms |
| P99 | 450ms | 1800ms |
| Max | 1200ms | 3500ms |
```

**Include visualizations:**
- Bar chart: Accuracy comparison
- Box plot: Latency distribution
- Confusion matrix: Error patterns

### Section 4: Cost Analysis

**Break down costs clearly:**

**Training Costs:**
```
OpenAI:
  - Data preparation: 1 hour (engineer time)
  - API upload: $0
  - Training: $18.50
  - Total: $18.50 + engineer time

QLoRA:
  - Data preparation: 2 hours (engineer time)
  - GPU rental: 2 hours × $0.60 = $1.20
  - Training: $1.20
  - Total: $1.20 + engineer time
```

**Inference Costs (per 1,000 requests):**
```
OpenAI: $0.15
QLoRA (self-hosted): $0.02
QLoRA (cloud-hosted): $0.05

Savings: 87% (self-hosted) or 67% (cloud-hosted)
```

**Break-Even Analysis:**
```
At 100,000 inferences:
  OpenAI: $18.50 + $15.00 = $33.50
  QLoRA: $1.20 + $2.00 = $3.20
  Savings: $30.30 (90%)

At 1,000,000 inferences:
  OpenAI: $18.50 + $150.00 = $168.50
  QLoRA: $1.20 + $20.00 = $21.20
  Savings: $147.30 (87%)
```

### Section 5: Deployment Considerations

**Create a comparison matrix:**

| Factor | OpenAI | QLoRA | Winner |
|--------|--------|-------|--------|
| Time to Deploy | 1 day | 1 week | OpenAI |
| Scalability | Auto | Manual | OpenAI |
| Latency | 215ms | 850ms | OpenAI |
| Cost (high volume) | High | Low | QLoRA |
| Data Privacy | External | On-premise | QLoRA |
| Customization | Limited | Full | QLoRA |
| Maintenance | None | High | OpenAI |
| Offline Capability | No | Yes | QLoRA |

### Section 6: Recommendations

**Provide clear decision framework:**

**Use OpenAI when:**
- ✅ Rapid prototyping (need demo in days)
- ✅ Low inference volume (<10K/day)
- ✅ No ML infrastructure or expertise
- ✅ Data privacy is not critical
- ✅ Need guaranteed uptime (SLA)
- ✅ Budget for ongoing API costs

**Use QLoRA when:**
- ✅ High inference volume (>100K/day)
- ✅ Data privacy is critical (HIPAA, GDPR)
- ✅ Have ML engineering team
- ✅ Long-term deployment (>6 months)
- ✅ Need offline/edge deployment
- ✅ Want full model control

**Hybrid Approach:**
- Start with OpenAI for MVP
- Collect real usage data
- Switch to QLoRA at scale
- Keep OpenAI as fallback

### Section 7: Lessons Learned

**Reflect on the process:**

**What worked well:**
- Both models improved significantly over baseline
- Evaluation framework was comprehensive
- Cost analysis revealed clear break-even point

**What was challenging:**
- QLoRA required more debugging
- Hyperparameter tuning took time
- Latency optimization needed iteration

**What would you do differently:**
- Start with smaller model for faster iteration
- Collect more diverse test cases
- Implement A/B testing with real users

---

## Lecture 7: Presentation Best Practices

### Storytelling Structure

**Act 1: The Problem (2 minutes)**
- Medical Q&A is hard for general LLMs
- Need to choose between OpenAI and QLoRA
- No clear guidance exists

**Act 2: The Experiment (5 minutes)**
- Describe dataset and approach
- Show training process for both
- Present evaluation framework

**Act 3: The Results (5 minutes)**
- Performance comparison
- Cost analysis
- Deployment considerations

**Act 4: The Recommendation (3 minutes)**
- Decision framework
- Use case examples
- Key takeaways

### Slide Design Principles

**1. One Message Per Slide**
❌ Bad: Slide with 5 different charts
✅ Good: One chart with clear title stating the finding

**2. Large, Readable Text**
- Minimum 24pt font
- High contrast (dark text on light background)
- Readable from 20 feet away

**3. Effective Visualizations**
- Bar charts for comparisons
- Line charts for trends
- Tables for exact numbers (keep small!)

**4. Avoid Code on Slides**
- Show architecture diagrams instead
- Use pseudocode if necessary
- Link to GitHub for full code

### Handling Questions

**Common Questions:**

**Q: "Why is QLoRA accuracy lower?"**
A: Smaller model (3B vs proprietary), less training data (2K vs 8K in our case), quantization trade-off. The 1.5% difference is acceptable given 87% cost savings.

**Q: "Can you make QLoRA faster?"**
A: Yes! Options: Better GPU, quantization optimization, batching, caching. We focused on cost, but latency can be improved.

**Q: "What about GPT-4?"**
A: GPT-4 would be more accurate but 10x more expensive. We chose GPT-4o-mini for fair cost comparison.

**Q: "Why not use both?"**
A: Great idea! Hybrid approach: OpenAI for critical queries, QLoRA for bulk processing.

---

## Key Takeaways for Students

### 1. There's No Universal Best Answer

The "best" approach depends on:
- Your constraints (budget, time, expertise)
- Your requirements (latency, privacy, volume)
- Your priorities (cost vs performance vs control)

### 2. Always Benchmark with YOUR Data

Academic papers show one thing, your production data may show another. Always test with your specific use case.

### 3. Consider Total Cost of Ownership

Don't just look at API costs. Consider:
- Engineering time
- Infrastructure costs
- Maintenance burden
- Opportunity cost

### 4. Start Simple, Optimize Later

Begin with OpenAI for rapid iteration. Once you validate the use case and have real traffic, consider switching to QLoRA for cost optimization.

### 5. The Landscape is Evolving

- New models released monthly
- Prices change
- New techniques emerge (e.g., QLoRA → GPTQ → AWQ)
- Stay updated and re-evaluate periodically

### 6. Document Everything

Your future self (and teammates) will thank you:
- Hyperparameters and why you chose them
- Training curves and metrics
- Cost breakdowns
- Lessons learned

### 7. Think Like a Business

Technical excellence matters, but business impact matters more:
- Will this save money?
- Will this improve user experience?
- Can we maintain this long-term?
- What's the ROI?

---

## Conclusion

This capstone project isn't just about learning two fine-tuning techniques. It's about developing the judgment to choose the right tool for the right job.

In your career, you'll face many such decisions:
- Cloud vs on-premise
- Build vs buy
- Optimize for speed vs cost vs quality

The framework you've built here—rigorous evaluation, cost analysis, trade-off assessment—applies to all of them.

**Good luck with your Fine-Tuning Showdown! 🚀**

