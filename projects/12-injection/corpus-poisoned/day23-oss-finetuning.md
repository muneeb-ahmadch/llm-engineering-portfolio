# Fine-Tuning Open-Source LLMs with QLoRA
## Comprehensive Lecture Notes

---

## Introduction

Welcome to this lecture on fine-tuning open-source Large Language Models (LLMs) using QLoRA! In this session, we'll explore how you can take a pre-trained model like Meta's Llama and adapt it to your specific needs - all without needing expensive hardware or massive computational resources.

### What We'll Cover

1. Understanding what "building your own model" really means
2. The challenge of fine-tuning large models
3. LoRA (Low-Rank Adaptation) technique
4. Quantization for memory efficiency
5. QLoRA: Combining both techniques
6. Practical implementation steps
7. Best practices and common pitfalls

---

## 1. What Does "Building Your Own Model" Mean?

When people in the industry say "we built our own model" or "we have our own LLM," they usually don't mean they trained a model from scratch. What they actually mean is that they took an existing open-source model and **fine-tuned** it for their specific use case.

### The Reality of "Building" Models

**What we're NOT doing:**
- Training a model from scratch (costs millions of dollars)
- Creating a new architecture
- Pre-training on massive web-scale datasets

**What we ARE doing:**
- Taking a pre-trained open-source model (like Llama 3.2)
- Fine-tuning it on our specific dataset
- Adapting it to perform well on our particular task

### Why This Approach Makes Sense

Think of it like this: If you wanted to become a doctor, you wouldn't start by reinventing all of medical science from scratch. You'd build on centuries of accumulated knowledge. Similarly, we build on the knowledge already embedded in pre-trained models.

**Example:** Bloomberg did train Bloomberg GPT from scratch, but this is extremely rare. Most companies find that by the time they finish training, frontier labs have released better models, making their investment less worthwhile.

### Popular Open-Source Base Models

1. **Meta's Llama** (most common in industry)
   - Llama 3.2: 1B, 3B parameters
   - Llama 3.1: 8B, 70B, 405B parameters
   - Open-source, commercially usable

2. **Google's Gemma**
   - 2B, 7B parameters
   - Good performance, easy to use

3. **Microsoft's Phi**
   - Phi-3: 3.8B parameters
   - Optimized for efficiency

4. **DeepSeek**
   - Various sizes
   - Strong performance

5. **Alibaba's Qwen**
   - Multilingual capabilities
   - No approval process needed

---

## 2. The Challenge: Memory Requirements

Before we dive into solutions, let's understand the problem we're solving.

### The Memory Problem

**Llama 3.2 with 3 billion parameters:**
- Each parameter is stored as a 32-bit floating-point number
- 32 bits = 4 bytes
- 3 billion × 4 bytes = 12 GB just to store the model
- Actual size: ~13 GB (with some overhead)

**But training requires even more memory:**
1. **Model weights:** 13 GB
2. **Gradients:** 13 GB (need to calculate how to adjust each weight)
3. **Optimizer states:** 26 GB (Adam optimizer keeps momentum and variance)
4. **Activations:** Variable (depends on batch size)

**Total:** ~50-60 GB of GPU memory just to train!

### Why This is a Problem

Most consumer GPUs have:
- RTX 3090: 24 GB
- RTX 4090: 24 GB
- Google Colab Free T4: 15 GB

**We can't fit the model for training!**

This is where QLoRA comes to the rescue.

---

## 3. LoRA: Low-Rank Adaptation

LoRA is the first trick that makes efficient fine-tuning possible. Let's break it down step by step.

### The Core Idea

Instead of training all 3 billion parameters, we:
1. **Freeze** all the original model weights (they don't change)
2. Add small **adapter matrices** alongside the frozen weights
3. Train only these small adapter matrices
4. The adapters "adapt" the frozen model's behavior

Think of it like this: Instead of rewriting an entire book, you're adding sticky notes with corrections and additions.

### How LoRA Works Mathematically

**Original layer:**
```
Output = Weight_Matrix × Input
```

**With LoRA:**
```
Output = Weight_Matrix × Input + (alpha × LoRA_A × LoRA_B) × Input
```

Where:
- `Weight_Matrix` is frozen (not trained)
- `LoRA_A` and `LoRA_B` are small trainable matrices
- `alpha` is a scaling factor

### Why Two Matrices (A and B)?

You might wonder: why not just one small matrix?

**The answer is about dimensions:**
- Original weight matrix: 3072 × 3072 (example)
- We can't directly add a small matrix to a large one
- Solution: Use two matrices that multiply together

**Example:**
- LoRA_A: 3072 × 32 (large × small)
- LoRA_B: 32 × 3072 (small × large)
- LoRA_A × LoRA_B: 3072 × 3072 (matches original!)

The "32" in the middle is called **r** (rank), and it's much smaller than 3072.

### Target Modules

Not all layers in the model need LoRA adapters. We typically target specific layers:

**Llama 3.2 Architecture:**
- 28 decoder layers (stacked)
- Each layer has:
  - Self-attention (Q, K, V, O projections)
  - MLP (Multi-Layer Perceptron)
  - Layer normalization

**Common strategy:**
1. **Start with attention layers:** q_proj, k_proj, v_proj, o_proj
2. **Add MLP if needed:** gate_proj, up_proj, down_proj
3. **Rarely:** LM head (output layer)

**Why attention layers?**
- They're responsible for learning which parts of input matter most
- Adapting them gives good results with fewer parameters
- Rule of thumb: Start here, expand if needed

### LoRA Hyperparameters

**r (rank):** The inner dimension of LoRA matrices
- Common values: 8, 16, 32, 64, 128, 256
- Higher r = more capacity = slower training = more memory
- **Start with:** 32 (good balance)

**alpha:** Scaling factor for LoRA updates
- Rule of thumb: alpha = 2 × r
- Example: r=32 → alpha=64
- This is empirically found to work well
- You can experiment, but doubling r is standard

**dropout:** Probability of randomly zeroing neurons
- Common values: 0.1 (10%) or 0.2 (20%)
- Helps prevent overfitting
- **Start with:** 0.1

### Memory Savings with LoRA

**Example calculation (r=32, targeting Q,K,V,O):**
- Each LoRA layer: ~200K parameters
- 28 layers × 4 attention heads = 112 LoRA matrices
- Total: ~18 million parameters
- At 32-bit: 18M × 4 bytes = 72 MB

**Compare to original:**
- Original: 3 billion parameters (13 GB)
- LoRA: 18 million parameters (72 MB)
- **We're training 0.6% of the parameters!**

---

## 4. Quantization: The Q in QLoRA

Quantization is the second trick that makes QLoRA so powerful.

### What is Quantization?

Quantization means reducing the precision of numbers used to store model weights.

**Analogy: Dimmer Switch**
- **32-bit float:** Infinite positions (smooth dimming)
- **8-bit:** 256 positions (fine steps)
- **4-bit:** 16 positions (coarse steps)

You're reducing how finely you can adjust each "knob" in the model.

### Precision Levels

**32-bit floating point (FP32):**
- Standard precision
- Range: ±3.4 × 10³⁸
- 4 bytes per number

**16-bit floating point (FP16):**
- Half precision
- Range: ±65,504
- 2 bytes per number

**8-bit integer (INT8):**
- 256 possible values
- 1 byte per number

**4-bit (NF4 - Normal Float 4):**
- 16 possible values
- 0.5 bytes per number
- Special encoding for neural networks

### How Much Information Do We Lose?

This is the surprising part: **Not as much as you'd think!**

**Empirical findings:**
- 8-bit quantization: ~1-2% performance drop
- 4-bit quantization: ~2-5% performance drop
- Much better than removing 75% of layers!

**Why does this work?**
- Neural networks learn more information than strictly necessary
- There's redundancy in the precision
- Like MP3 compression for audio - you lose some quality but it's still very usable

### Memory Savings with Quantization

**Llama 3.2 (3B parameters):**
- 32-bit: 13 GB
- 8-bit: 3.6 GB (72% reduction)
- 4-bit: 2.2 GB (83% reduction)

**This is huge!** We can now fit the model in Google Colab's free T4 GPU (15 GB).

### Important Clarifications

**What gets quantized:**
- The **base model** weights get quantized
- The **LoRA adapters** stay at full 32-bit precision
- This is important for training stability

**When quantization happens:**
- During loading (on-the-fly)
- Not during training (LoRA adapters are trained at full precision)
- Can quantize again for deployment if needed

### NF4 Quantization

**NF4 (Normal Float 4)** is a special 4-bit format designed for neural networks:
- Maps 16 values to a normal distribution
- Optimized for the typical range of neural network weights
- Better than naive 4-bit quantization
- This is what QLoRA uses

---

## 5. QLoRA: Putting It All Together

Now we combine both techniques!

### The QLoRA Recipe

1. **Load base model with 4-bit quantization**
   - Llama 3.2: 13 GB → 2.2 GB

2. **Add LoRA adapters (not quantized)**
   - Add 72 MB of trainable parameters

3. **Total memory for model: 2.27 GB**
   - Fits easily in 15 GB GPU!

4. **Train only the LoRA adapters**
   - Base model stays frozen
   - Much faster training

5. **Save only the LoRA weights**
   - 72 MB file to save/share
   - Can load on top of any quantized base model

### Why QLoRA is Revolutionary

**Before QLoRA:**
- Needed 50+ GB GPU to fine-tune 3B model
- Required expensive hardware ($10K+ GPU)
- Training took hours/days

**With QLoRA:**
- Need only 15 GB GPU (free on Colab!)
- Can use consumer hardware
- Training takes 30 minutes to few hours

### QLoRA Configuration

**Typical setup:**
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Enable 4-bit quantization
    bnb_4bit_quant_type="nf4",            # Use NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in bfloat16
    bnb_4bit_use_double_quant=True        # Double quantization
)
```

**LoRA configuration:**
```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=32,                                  # Rank
    lora_alpha=64,                         # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,                      # Dropout
    bias="none",                           # Don't adapt biases
    task_type="CAUSAL_LM"                  # Task type
)
```

---

## 6. Base Model vs Chat/Instruct Model

This is a crucial concept that often confuses beginners.

### Two Types of Models

**Base Model:**
- Trained to predict next token
- Input: Any text sequence
- Output: Most likely continuation
- Example: GPT-2, GPT-3 (original)

**Chat/Instruct Model:**
- Fine-tuned from base model
- Trained with special format (system, user, assistant)
- Expects structured conversation
- Example: GPT-3.5-turbo, GPT-4, Llama-3-Instruct

### When to Use Each

**Use Base Model when:**
- You have a single, consistent task
- You want simple prompt → completion
- You're teaching a specialized skill
- Example: "Predict price from description"

**Use Chat/Instruct Model when:**
- You need conversational ability
- Instructions vary significantly
- You want system prompts
- Example: Customer service chatbot

### Data Format Differences

**Base Model:**
```python
{
    "prompt": "What does this cost? [description] Price is $",
    "completion": "149"
}
```

**Chat Model:**
```python
{
    "messages": [
        {"role": "system", "content": "You are a pricing expert"},
        {"role": "user", "content": "What does this cost? [description]"},
        {"role": "assistant", "content": "This costs $149"}
    ]
}
```

### Which is Better for Fine-Tuning?

**Rule of thumb:**
- **Base model:** Better for specialized, single-task problems
- **Instruct model:** Better for varied instructions and conversations

**Always experiment with both!** The best choice depends on your specific use case.

---

## 7. Training Hyperparameters

Beyond LoRA-specific settings, you need to configure general training parameters.

### Epochs

**Definition:** One complete pass through your entire training dataset

**How many epochs?**
- Start with: 1-3 epochs
- More epochs = more training = risk of overfitting
- Monitor validation loss to decide

**Why multiple epochs?**
- Each training step only makes a small adjustment
- Data is batched differently each epoch (random sampling)
- Model gradually improves with each pass

### Batch Size

**Definition:** Number of examples processed together in one training step

**How to choose:**
- **Rule of thumb:** As large as fits in GPU memory
- Common values: 4, 8, 16, 32
- Larger batch = faster training, more stable gradients
- Smaller batch = slower training, more noise (can help generalization)

**Gradient Accumulation:**
- If batch size is too small, accumulate gradients over multiple steps
- Simulates larger batch size without memory cost
- Example: batch_size=4, gradient_accumulation_steps=4 → effective batch of 16

### Learning Rate

**Definition:** How big a step to take when updating weights

**Typical values:**
- For fine-tuning: 1e-4 to 5e-4 (0.0001 to 0.0005)
- Too high: Training unstable, loss explodes
- Too low: Training too slow, might not converge

**Learning Rate Scheduling:**
- **Constant:** Same rate throughout
- **Linear decay:** Gradually decrease
- **Cosine:** Smooth decrease following cosine curve
- **Warmup:** Start low, increase, then decrease

**Recommended:** Linear or cosine with warmup

### Dropout

**Definition:** Randomly "turn off" neurons during training

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

### Data Format

**For base models:**
```python
{
    "prompt": "Question or context ending with a clear signal",
    "completion": "Expected output"
}
```

**Example (price prediction):**
```python
{
    "prompt": "What does this cost to the nearest dollar? \n[Product: Gaming Mouse]\n[Brand: Logitech]\n[Description: Wireless gaming mouse with RGB]\nPrice is $",
    "completion": "79"
}
```

### Data Quality Matters

**Good data characteristics:**
- Consistent format across all examples
- Clear signal for where completion should start
- Balanced distribution (not all same answer)
- Clean, accurate labels
- Sufficient quantity (1000+ examples ideal)

**Common issues:**
- Inconsistent formatting
- Noisy labels
- Class imbalance
- Too short/too long examples

### Train/Validation/Test Split

**Standard split:**
- Training: 80% (used for training)
- Validation: 10% (used during training to monitor)
- Test: 10% (held out for final evaluation)

**Important:** Never let model see test data during training!

### Token Length Management

**Problem:** Variable-length inputs waste memory

**Solution:**
1. Tokenize all examples
2. Plot distribution of lengths
3. Choose cutoff (e.g., 95th percentile)
4. Truncate longer examples
5. Pad shorter examples

**Example:**
- 95% of data: <110 tokens
- Set max_length=110
- Only 5% gets truncated
- Much more efficient training!

---

## 9. The Training Process

Let's walk through what actually happens during training.

### Step-by-Step Training

**1. Load Quantized Base Model**
```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config,
    device_map="auto"
)
```

**2. Add LoRA Adapters**
```python
from peft import get_peft_model

model = get_peft_model(model, lora_config)
```

**3. Prepare Dataset**
```python
from datasets import load_dataset

dataset = load_dataset("your_dataset")
```

**4. Configure Trainer**
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    peft_config=lora_config,
    max_seq_length=110,
    args=training_args
)
```

**5. Train!**
```python
trainer.train()
```

**6. Save LoRA Weights**
```python
model.save_pretrained("./my_lora_model")
```

### What Happens During Training

**Each training step:**
1. **Forward Pass:** Input → Model → Prediction
2. **Calculate Loss:** How wrong was the prediction?
3. **Backward Pass:** Calculate gradients (how to improve)
4. **Update Weights:** Adjust LoRA matrices
5. **Repeat** for next batch

**Each epoch:**
- Go through all training data
- Evaluate on validation set
- Log metrics (loss, learning rate, etc.)
- Save checkpoint if best so far

### Monitoring Training

**Key metrics to watch:**
- **Training Loss:** Should steadily decrease
- **Validation Loss:** Should decrease (watch for overfitting!)
- **Learning Rate:** Changes based on schedule
- **GPU Memory:** Should stay under limit
- **Time per step:** Helps estimate total time

**Warning signs:**
- Validation loss increasing while training loss decreases → Overfitting!
- Loss not decreasing → Learning rate too low or data issues
- Loss exploding (NaN) → Learning rate too high

---

## 10. Evaluation and Iteration

Training is just the beginning. Evaluation tells you if it worked!

### Evaluation Metrics

**For regression tasks (e.g., price prediction):**
- **MAE (Mean Absolute Error):** Average difference
- **RMSE (Root Mean Squared Error):** Penalizes large errors
- **R² Score:** How much variance explained

**For classification tasks:**
- **Accuracy:** Percentage correct
- **Precision:** Of predicted positives, how many correct?
- **Recall:** Of actual positives, how many found?
- **F1 Score:** Harmonic mean of precision and recall

**For generation tasks:**
- **BLEU Score:** Overlap with reference text
- **ROUGE Score:** Recall-oriented overlap
- **Human Evaluation:** Often most important!

### Comparing to Baseline

**Always compare to:**
1. **Base model (no fine-tuning):** How much did we improve?
2. **Simple baseline:** (e.g., always predict mean)
3. **Previous best:** Are we doing better than before?

**Example:**
- Simple baseline: MAE = $106
- Base Llama 3.2: MAE = $110 (worse than baseline!)
- After QLoRA fine-tuning: MAE = $35 (huge improvement!)

### Iteration Cycle

**1. Train with initial hyperparameters**
**2. Evaluate on test set**
**3. Analyze errors:**
   - Where does model fail?
   - What patterns in mistakes?
   - Is it overfitting or underfitting?

**4. Adjust:**
   - Change hyperparameters (r, learning rate, epochs)
   - Improve data quality
   - Add more data
   - Try different target modules

**5. Repeat until satisfied**

---

## 11. Best Practices

### Do's

✅ **Start simple:** r=32, alpha=64, target attention layers only
✅ **Monitor validation loss:** Stop if it starts increasing
✅ **Save checkpoints:** Don't lose progress
✅ **Version control:** Track what you tried
✅ **Document everything:** What worked, what didn't
✅ **Test thoroughly:** Use held-out test set
✅ **Compare to baselines:** Know your improvement
✅ **Experiment systematically:** Change one thing at a time

### Don'ts

❌ **Don't skip data quality:** Garbage in, garbage out
❌ **Don't overtrain:** Watch for overfitting
❌ **Don't ignore validation loss:** It's your early warning
❌ **Don't use test set for tuning:** Keep it truly held-out
❌ **Don't forget to quantize:** Wastes memory
❌ **Don't use too high learning rate:** Causes instability
❌ **Don't give up too early:** Fine-tuning takes experimentation

---

## 12. Common Pitfalls and Solutions

### Pitfall 1: Out of Memory (OOM)

**Symptoms:** CUDA out of memory error

**Solutions:**
- Reduce batch size
- Reduce max_sequence_length
- Use gradient accumulation
- Reduce r (LoRA rank)
- Use 4-bit instead of 8-bit quantization

### Pitfall 2: Model Not Learning

**Symptoms:** Loss not decreasing

**Solutions:**
- Increase learning rate
- Check data quality
- Increase number of epochs
- Increase r (model capacity)
- Check if data format is correct

### Pitfall 3: Overfitting

**Symptoms:** Training loss decreases, validation loss increases

**Solutions:**
- Reduce number of epochs
- Increase dropout
- Add more training data
- Use weight decay
- Reduce model capacity (lower r)

### Pitfall 4: Unstable Training

**Symptoms:** Loss jumps around, sometimes becomes NaN

**Solutions:**
- Decrease learning rate
- Use gradient clipping
- Check for data issues (extreme values)
- Use warmup steps

### Pitfall 5: Slow Training

**Symptoms:** Training takes too long

**Solutions:**
- Increase batch size
- Reduce max_sequence_length
- Use fewer target modules
- Use gradient checkpointing
- Upgrade hardware

---

## 13. Deployment Considerations

After training, you need to deploy your model.

### Saving Your Model

**Two components to save:**
1. **Base model:** Can reference the original (e.g., "meta-llama/Llama-3.2-3B")
2. **LoRA adapters:** Your 72 MB of trained weights

**Saving:**
```python
# Save LoRA adapters
model.save_pretrained("./my_model")

# Push to Hugging Face Hub
model.push_to_hub("username/my-model")
```

### Loading for Inference

**Loading:**
```python
from peft import PeftModel

# Load base model (quantized)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=bnb_config
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./my_model")
```

### Inference Speed

**Factors affecting speed:**
- Model size (3B vs 70B)
- Quantization (4-bit faster than 32-bit)
- Hardware (GPU vs CPU)
- Batch size (process multiple at once)

**Typical speeds (3B model, 4-bit, T4 GPU):**
- Single inference: ~1-2 seconds
- Batch of 10: ~3-4 seconds
- Much faster than training!

### Deployment Options

**1. Local Deployment:**
- Run on your machine
- Full control, no API costs
- Need GPU for reasonable speed

**2. Cloud API:**
- Deploy on AWS/GCP/Azure
- Scalable, pay per use
- Example: AWS SageMaker, GCP Vertex AI

**3. Hugging Face Spaces:**
- Free hosting for demos
- Easy to set up
- Good for prototypes

**4. Edge Devices:**
- For very small models (1B-3B)
- Mobile phones, IoT devices
- Need aggressive quantization

---

## 14. Real-World Applications

Let's look at practical use cases where QLoRA shines.

### Use Case 1: Domain-Specific Q&A

**Scenario:** Medical diagnosis assistant

**Why QLoRA:**
- Need specialized medical knowledge
- Privacy concerns (can't use API)
- Cost-effective for high volume

**Approach:**
- Base: Llama 3.2 3B
- Data: Medical Q&A pairs
- Fine-tune with QLoRA
- Deploy locally in hospitals

### Use Case 2: Custom Code Generation

**Scenario:** Generate code in company's specific framework

**Why QLoRA:**
- Frontier models don't know internal frameworks
- Need consistent style/patterns
- Frequent updates as framework evolves

**Approach:**
- Base: CodeLlama or similar
- Data: Internal code examples
- Fine-tune for company patterns
- Deploy as IDE plugin

### Use Case 3: Specialized Translation

**Scenario:** Translate technical documents with domain terminology

**Why QLoRA:**
- Generic translation misses technical terms
- Need consistency in terminology
- Cost savings vs API

**Approach:**
- Base: Multilingual model
- Data: Parallel technical documents
- Fine-tune for domain
- Deploy as internal tool

### Use Case 4: Content Moderation

**Scenario:** Detect policy violations in user content

**Why QLoRA:**
- Company-specific policies
- Need fast, local processing
- Privacy requirements

**Approach:**
- Base: Llama 3.2 3B
- Data: Labeled examples of violations
- Fine-tune for policy detection
- Deploy in content pipeline

---

## 15. Advanced Topics (Brief Overview)

These are beyond the scope of this lecture but worth knowing about.

### Multi-LoRA

**Concept:** Train multiple LoRA adapters for different tasks

**Use case:** One base model, swap adapters for different tasks
- Adapter 1: Customer support
- Adapter 2: Technical writing
- Adapter 3: Code generation

### LoRA Merging

**Concept:** Combine multiple LoRA adapters

**Use case:** Merge adapters trained on different datasets

### QA-LoRA

**Concept:** Quantization-aware LoRA training

**Benefit:** Even better performance with quantization

### DoRA (Weight-Decomposed LoRA)

**Concept:** Decompose weights differently

**Benefit:** Better performance than standard LoRA

---

## Summary

### Key Takeaways

1. **QLoRA combines two tricks:**
   - LoRA: Train small adapters instead of full model
   - Quantization: Reduce precision to save memory

2. **Memory savings are dramatic:**
   - 50+ GB → 2.2 GB for training
   - Can use free Google Colab!

3. **Training only 0.6% of parameters:**
   - 18 million vs 3 billion
   - Much faster training

4. **Choose base vs instruct model:**
   - Base: Single-task, specialized
   - Instruct: Multi-task, conversational

5. **Key hyperparameters:**
   - r=32, alpha=64 (good starting point)
   - Target attention layers first
   - Learning rate: 1e-4 to 5e-4
   - Dropout: 0.1

6. **Data quality matters most:**
   - Clean, consistent format
   - Sufficient quantity (1000+)
   - Proper train/val/test split

7. **Monitor and iterate:**
   - Watch validation loss
   - Compare to baselines
   - Experiment systematically

### What You Can Do Now

With QLoRA, you can:
- ✅ Fine-tune 3B parameter models on free hardware
- ✅ Create specialized models for your domain
- ✅ Deploy models locally (no API costs)
- ✅ Maintain data privacy
- ✅ Iterate quickly (30 min - 2 hour training)

### Next Steps

1. **Try it yourself:** Google Colab + Llama 3.2
2. **Start simple:** Use example datasets first
3. **Experiment:** Try different hyperparameters
4. **Build something:** Apply to your own use case
5. **Share:** Contribute to open-source community

---

## Additional Resources

### Papers

- **LoRA:** "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
  - https://arxiv.org/abs/2106.09685

- **QLoRA:** "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
  - https://arxiv.org/abs/2305.14314

### Documentation

- **Hugging Face PEFT:** https://huggingface.co/docs/peft
- **Hugging Face TRL:** https://huggingface.co/docs/trl
- **Transformers:** https://huggingface.co/docs/transformers
- **Bits and Bytes:** https://github.com/TimDettmers/bitsandbytes

### Tutorials

- **Hugging Face Course:** https://huggingface.co/learn
- **PEFT Examples:** https://github.com/huggingface/peft/tree/main/examples
- **Google Colab Notebooks:** Search "QLoRA fine-tuning tutorial"

### Community

- **Hugging Face Forums:** https://discuss.huggingface.co/
- **Reddit:** r/LocalLLaMA, r/MachineLearning
- **Discord:** Hugging Face, EleutherAI
- **Twitter/X:** Follow @huggingface, @Tim_Dettmers

---

## Conclusion

Fine-tuning open-source LLMs with QLoRA has democratized access to custom AI models. What once required millions of dollars and specialized hardware can now be done on free Google Colab in an afternoon.

The key is understanding the trade-offs:
- Smaller models vs frontier models
- Training time vs performance
- Memory vs capacity
- Generalization vs specialization

With QLoRA, you have a powerful tool to create specialized models that excel at your specific tasks. The barrier to entry has never been lower.

**Now go build something amazing!**

---

**End of Lecture Notes**

