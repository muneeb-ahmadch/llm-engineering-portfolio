# Experiment 04: chunks to judge

Config: `bge-small` · chunk 1000 · top-5 · relevance=`all`. Regenerate any time by re-running `show_chunks.py`.

Open this in the Markdown **preview** (right-click → *Open Preview*, or ⌘K V). Each chunk is a collapsible block, click a `#N` heading to fold it away once you've judged it. Then record your verdicts in [`worksheet.md`](worksheet.md).

---

## What is overfitting in a neural network?

**Ruler verdict:** ❌ MISS, scored 0 · coverage 33% · category `concept`  
**Keywords demanded:** `Overfitting` · `Training accuracy` · `Validation accuracy`  
**Reference answer** (ground truth, not scored): Overfitting is when a model learns the training data too closely, including its noise, so it generalizes poorly, visible when training accuracy is much higher than validation accuracy.

> **Your job:** ignore the ✓/✗ below. Does any chunk actually *answer* the question, in whatever words? Note the rank of the one that does (or that none does).

<details open>
<summary><b>#1</b> · sim 0.751 · <code>day23-oss-finetuning.md</code> · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

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
<summary><b>#2</b> · sim 0.717 · <code>day26-deep-learning.md</code> · Overfitting ✗ · Training accuracy ✗ · Validation accuracy ✗</summary>

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
<summary><b>#3</b> · sim 0.709 · <code>day27-deep-learning-2.md</code> · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

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
<summary><b>#4</b> · sim 0.706 · <code>day27-deep-learning-2.md</code> · Overfitting ✓ · Training accuracy ✗ · Validation accuracy ✗</summary>

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
<summary><b>#5</b> · sim 0.701 · <code>day26-deep-learning.md</code> · Overfitting ✗ · Training accuracy ✗ · Validation accuracy ✗</summary>

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


---

## What is prompt caching?

**Ruler verdict:** ✅ PASS, first relevant at rank 1 · coverage 100% · category `concept`  
**Keywords demanded:** `prompt caching` · `reuse` · `processing`  
**Reference answer** (ground truth, not scored): Prompt caching saves the processing work for a frequently repeated prompt so the provider reuses it instead of recomputing from scratch each time, which saves money and latency.

> **Your job:** ignore the ✓/✗ below. Does any chunk actually *answer* the question, in whatever words? Note the rank of the one that does (or that none does).

<details open>
<summary><b>#1</b> · sim 0.710 · <code>day13-multimodal.md</code> · prompt caching ✓ · reuse ✓ · processing ✓ · **◀ all keywords**</summary>

```text
ice bot that always includes your company's 10-page policy document in the system prompt. Instead of processing that document from scratch every time, caching lets the provider remember it.

### How Different Providers Handle Caching

**OpenAI (Automatic):**
- Caching happens automatically
- You pay about 5x less for cached tokens
- The beginning of your prompt must match exactly

**Anthropic (Explicit):**
- You must tell it what to cache
- 25% more expensive to "prime" the cache
- 10x cheaper when you reuse cached content
- Bigger overall savings if you plan for it

**Gemini:**
- Supports both automatic and explicit modes

### The Golden Rule of Prompt Caching

**Put static content FIRST, variable content LAST.**

❌ Wrong approach:
```
Today's date is January 13, 2026.
[Large context document]
User question: How do I return an item?
```

✅ Correct approach:
```
[Large context document]
User question: How do I return an item?
Today's date is January 13, 2026.
```

Why? Because caching
```
</details>

<details open>
<summary><b>#2</b> · sim 0.667 · <code>day13-multimodal.md</code> · prompt caching ✗ · reuse ✗ · processing ✗</summary>

```text
te is January 13, 2026.
```

Why? Because caching matches from the beginning of the prompt. If the first thing that changes (like today's date), nothing gets cached!

---

## Part 3: LLM Conversations

### Making Two LLMs Talk to Each Other

Here's something fun: you can set up conversations between different LLMs! This might sound silly, but it's actually a great way to understand how the message format works.

**The setup:**
```python
gpt_system = "You are argumentative. You disagree with everything in a snarky way."
claude_system = "You are polite and courteous. You try to find common ground and calm things down."

gpt_messages = [{"role": "user", "content": "Hi there"}]
claude_messages = [{"role": "user", "content": "Hi"}]
```

**The conversation loop:**
```python
for i in range(5):
    # GPT responds
    gpt_response = call_gpt(gpt_messages)
    
    # Add GPT's response to Claude's messages as if a user said it
    claude_messages.append({"role": "user", "content": gpt_response})
```
</details>

<details open>
<summary><b>#3</b> · sim 0.653 · <code>day13-multimodal.md</code> · prompt caching ✗ · reuse ✗ · processing ✗</summary>

```text
xample: If the customer says "I'm looking for a hat", you could reply 
"Wonderful! We have lots of hats, including several on sale!"
```

**Multi-Shot Prompting:**
Give multiple examples:
```
You are a customer service assistant.

Example 1:
Customer: "Where's my order?"
You: "I'd be happy to help track your order. Could you provide your order number?"

Example 2:
Customer: "I want to return something"
You: "Of course! Our return policy allows returns within 30 days..."
```

More examples = more consistent, predictable behavior.

### Dynamic Context Injection

This is where things get really powerful. Instead of cramming everything into the system prompt, you add relevant information based on what the user asks:

```python
def get_system_prompt(user_message):
    base_prompt = "You are a helpful store assistant."
    
    if "belt" in user_message.lower():
        base_prompt += "\nImportant: We don't sell belts. Suggest other items."
    
    if "return" in user_message.lower():
```
</details>

<details open>
<summary><b>#4</b> · sim 0.641 · <code>day13-multimodal.md</code> · prompt caching ✗ · reuse ✗ · processing ✗</summary>

```text
if "return" in user_message.lower():
        base_prompt += "\nReturn policy: 30 days, receipt required..."
    
    return base_prompt
```

This is the foundation of **RAG (Retrieval Augmented Generation)**! The fancy versions use semantic search instead of keyword matching, but the core idea is the same: dynamically add relevant context to improve answers.

---

## Part 6: Tool Calling (Function Calling)

### What Are Tools?

Tools give LLMs the ability to do things beyond generating text:
- Look up information in databases
- Make calculations
- Book appointments
- Call external APIs
- Execute code

### The Big Revelation: No Magic!

Here's the thing that confuses many people: **LLMs don't actually run your code.** They only generate tokens. So how does tool calling work?

**The real process:**

1. **You** tell the LLM what tools are available (via JSON)
2. The LLM responds with: "Please run tool X with arguments Y"
3. **You** actually run the function in your code
4. **You**
```
</details>

<details open>
<summary><b>#5</b> · sim 0.635 · <code>day13-multimodal.md</code> · prompt caching ✓ · reuse ✓ · processing ✓ · **◀ all keywords**</summary>

```text
.content)
```

The beauty of LiteLLM is that you can easily switch providers:
- `openai/gpt-4o-mini` - OpenAI
- `anthropic/claude-3-5-sonnet` - Anthropic
- `bedrock/anthropic.claude-v2` - AWS Bedrock
- `azure/my-deployment` - Azure OpenAI

You just change the model string and LiteLLM handles the rest!

### Cost Tracking

One of LiteLLM's best features is built-in cost tracking:
```python
from litellm import completion_cost

response = completion(model="openai/gpt-4o-mini", messages=[...])
cost = completion_cost(response)
print(f"This call cost: ${cost:.6f}")
```

This is invaluable when building production systems where you need to monitor costs per user or per request.

---

## Part 2: Prompt Caching

### What is Prompt Caching?

When you send the same (or similar) prompts to an LLM repeatedly, prompt caching allows the provider to reuse some of the processing work, which saves you money.

**Example scenario:** You have a customer service bot that always includes your company's 10-pag
```
</details>


---

## What does reranking do in a RAG pipeline?

**Ruler verdict:** ✅ PASS, first relevant at rank 4 · coverage 100% · category `concept`  
**Keywords demanded:** `Reranking` · `reorder` · `relevance`  
**Reference answer** (ground truth, not scored): Reranking is a second pass that reorders an initially retrieved set of chunks by relevance to the query, often with a stronger model, so the most relevant passages rise to the top before being sent to the generator.

> **Your job:** ignore the ✓/✗ below. Does any chunk actually *answer* the question, in whatever words? Note the rank of the one that does (or that none does).

<details open>
<summary><b>#1</b> · sim 0.750 · <code>day16-rag-evals.md</code> · Reranking ✗ · reorder ✗ · relevance ✗</summary>

```text
chunks1 = fetch_unranked(question)
    chunks2 = fetch_unranked(rewritten)
    
    # Merge (remove duplicates)
    merged = merge_chunks(chunks1, chunks2)
    
    # Rerank the merged set
    reranked = rerank(question, merged)
    
    # Keep top 10
    return reranked[:10]
```

**Why This Works:**
- Original query: Good for direct matches
- Rewritten query: Good for clarified intent
- Merge: Best of both worlds
- Rerank: LLM filters out noise

### Multiprocessing for Speed

**The Problem:**
Processing 76 documents with LLM calls takes ~10 minutes serially.

**The Solution:**
```python
from multiprocessing import Pool

def process_document(doc):
    # LLM call to chunk document
    return chunks

# Process in parallel
with Pool(processes=5) as pool:
    all_chunks = pool.map(process_document, documents)
```

**Result:**
5x faster! (10 minutes → 2 minutes)

**Caution:**
Watch for rate limits. Start with 3-5 workers, increase if stable.

## Part 9: Putting It All Together

Let's s
```
</details>

<details open>
<summary><b>#2</b> · sim 0.736 · <code>day16-rag-evals.md</code> · Reranking ✓ · reorder ✗ · relevance ✗</summary>

```text
r than expected)

**Next Steps:**
- Implement hierarchical RAG for spanning questions
- Add aggregation summaries for holistic questions

## Part 11: Best Practices and Lessons Learned

### 1. Always Evaluate First

**Don't guess, measure!**

Before implementing any advanced technique, establish your baseline metrics. Otherwise, you won't know if your changes helped.

**Example:**
"I think reranking will help" → Measure → MRR improved by 0.08 → Keep it!
"I think query expansion will help" → Measure → MRR improved by 0.04 → Keep it!

### 2. Iterate Scientifically

**Change one thing at a time.**

If you change chunking, encoder, and reranking all at once, you won't know which change made the difference.

**Good process:**
1. Measure baseline
2. Change chunk size
3. Measure again
4. Keep or revert
5. Change encoder
6. Measure again
7. Keep or revert
... and so on

### 3. Don't Ignore Simple Solutions

**Prompt engineering is often overlooked.**

People get excited about fancy techniques
```
</details>

<details open>
<summary><b>#3</b> · sim 0.729 · <code>day16-rag-evals.md</code> · Reranking ✗ · reorder ✗ · relevance ✗</summary>

```text
able.

## Part 9: Putting It All Together

Let's see the complete advanced RAG pipeline:

### Ingest Pipeline (Advanced)

```python
# 1. Load documents
documents = load_documents("knowledge_base/")

# 2. Semantic chunking with LLM
chunks = []
for doc in documents:
    doc_chunks = llm_semantic_chunk(doc)  # LLM call
    chunks.extend(doc_chunks)

# 3. Generate embeddings
texts = [chunk.page_content for chunk in chunks]
embeddings = openai.embeddings.create(
    model="text-embedding-3-large",
    input=texts
).data

# 4. Store in Chroma
collection.add(
    embeddings=[e.embedding for e in embeddings],
    documents=texts,
    metadatas=[c.metadata for c in chunks],
    ids=[str(i) for i in range(len(chunks))]
)
```

### Query Pipeline (Advanced)

```python
def answer_question(question, history):
    # 1. Query rewriting
    rewritten = rewrite_query(question, history)
    
    # 2. Query expansion (retrieve for both)
    chunks1 = retrieve(question, k=20)
    chunks2 = retrieve(rewritt
```
</details>

<details open>
<summary><b>#4</b> · sim 0.725 · <code>day16-rag-evals.md</code> · Reranking ✓ · reorder ✓ · relevance ✓ · **◀ all keywords**</summary>

```text
llm.complete(
    prompt,
    response_format=Chunks
)
```

This ensures consistent, parseable output!

### Reranking Implementation

**The Prompt:**
```python
system_prompt = """You are a document ranker.
You are provided with a question and a list of chunks.
Rank the chunks by relevance to the question.
Reply only with a list of ranked chunk IDs."""

user_prompt = f"""Question: {question}

Chunks:
{numbered_chunks}

Order all chunks from most to least relevant."""
```

**Structured Output:**
```python
class RankOrder(BaseModel):
    order: list[int]  # [4, 1, 7, 2, ...]

# LLM returns ranked order
response = llm.complete(
    messages,
    response_format=RankOrder
)

# Reorder chunks
reranked = [chunks[i] for i in response.order]
```

**Example:**
```
Original order: [0, 1, 2, 3, 4]
LLM returns: [4, 1, 0, 2, 3]
Reranked chunks: [chunks[4], chunks[1], chunks[0], chunks[2], chunks[3]]
```

The chunk that was 5th is now 1st!

### Query Rewriting Implementation

**The Prompt:**
```pytho
```
</details>

<details open>
<summary><b>#5</b> · sim 0.721 · <code>day16-rag-evals.md</code> · Reranking ✓ · reorder ✓ · relevance ✓ · **◀ all keywords**</summary>

```text
and deduplicate → rerank.

**When to Use:**
- Low recall (missing relevant documents)
- Ambiguous queries
- Multiple ways to phrase the same question

**Our Approach:**
We'll use a simpler version: retrieve with both original and rewritten queries, then merge.

### Technique 7: Reranking

**What:** Use an LLM to reorder retrieved chunks by relevance.

**The Process:**
1. Retrieve many chunks (e.g., 20)
2. Ask LLM: "Given this question, rank these chunks by relevance"
3. LLM returns: [4, 1, 7, 2, ...] (chunk IDs in order)
4. Reorder chunks
5. Keep top 10 for final answer

**Why This Works:**
Vector similarity is good but not perfect. An LLM can understand:
- Semantic relevance
- Contextual importance
- Subtle connections

**Example:**
```
Question: "Who went to Manchester University?"

Initial retrieval (by vector similarity):
1. Chunk about London universities
2. Chunk about Jessica Liu's education (mentions Manchester)
3. Chunk about university partnerships
4. Chunk about Manchester
```
</details>


---
