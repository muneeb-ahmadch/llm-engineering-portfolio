# Day 16: RAG Evaluations & Advanced Techniques - Comprehensive Lecture Notes

## Introduction

Welcome to Day 16! Today we're taking a major step forward in our RAG journey. In Day 15, we built a working RAG pipeline using LangChain and Chroma. Today, we're going to learn how to evaluate whether our RAG system actually works well, and then we'll explore advanced techniques to make it production-ready.

Think of it this way: Building a RAG system is like building a car. Day 15 was about assembling the car and getting it to run. Today is about testing how fast it goes, how well it handles, and then tuning the engine to make it race-ready.

## Part 1: Understanding the RAG Reality

### What Makes RAG Great

Let's start by appreciating what we've built. RAG (Retrieval Augmented Generation) has some amazing qualities:

**1. Speed of Development**
Remember how quickly we built our InsureElm knowledge worker in Day 15? It took us just minutes to have a working system that could answer questions about employees, products, and services. Compare this to fine-tuning a model, which takes hours or days and significant computational resources.

**2. Scalability**
Our current system has 76 documents. What if we had 7,600 documents? Or 76,000? With RAG, we just ingest more documents into our vector store. The query time remains fast because we're still only retrieving the top-k most relevant chunks. This scalability is one of RAG's superpowers.

**3. Context Efficiency**
Instead of stuffing our entire knowledge base into the LLM's context window (which would be expensive and slow), we intelligently select only the most relevant pieces. This saves money and improves response quality.

**4. Focus and Relevance**
By retrieving only relevant chunks, we avoid polluting the LLM's context with irrelevant information that might confuse it or lead to hallucinations.

### The Not-So-Great Reality of RAG

Now for the honest truth that many tutorials won't tell you:

**RAG is Basically a Hack**

Let me explain. Transformers (the architecture behind LLMs) have attention mechanisms specifically designed to figure out what parts of the input are important. They've been trained on trillions of tokens to do this well. 

But we have a problem: we have too much information, and we don't want to waste our powerful (and expensive) transformer's time processing everything. So what do we do? We use RAG—a vector similarity search—as a shortcut to say "ah, we can probably zoom in on this piece and throw out the rest."

Notice all the "probably" language here. That's because RAG is fundamentally a performance optimization trick, not a rigorous solution.

**It's Very Empirical (Trial and Error)**

Students often ask me: "What chunking strategy should I use?" or "Which encoder is best for my use case?" 

The honest answer? I don't know until you try it.

Sure, I can make educated guesses based on experience, but RAG performance depends heavily on:
- Your specific documents
- The types of questions users ask
- The domain you're working in
- The quality of your test data

There's no universal "best" approach. You have to experiment and measure.

**Unexpected Failures (Whack-a-Mole)**

You might build a RAG system that works great for most questions, then someone asks "Who won the prestigious award?" and it returns "I don't know." You investigate and find that your chunking strategy split the person's name from the award information into different chunks.

You fix that problem by adjusting your chunking. Great! But now a different question starts failing. You fix that, and another issue pops up. This is the "whack-a-mole" nature of RAG optimization.

**Example: The Maxine Thompson Problem**

In our Day 15 system, when we asked "Who won the IoT award?", the system found chunks mentioning "Maxine" but not her last name "Thompson." The chunk boundaries split her full name. This is a classic RAG failure mode.

## Part 2: Why Evaluations Are Critical

Given that RAG is empirical and prone to unexpected failures, how do we make it reliable? The answer is **scientific evaluation**.

### The Problem Without Evaluations

Imagine you make a change to your RAG system—maybe you adjust the chunk size from 1000 to 500 characters. How do you know if this helped or hurt?

Without evaluations, you might:
- Test with a few hand-picked questions (biased)
- Rely on "it feels better" (not quantitative)
- Deploy and hope for the best (risky)

This is not how we build production systems.

### The Solution: Evaluation Framework

We need three things:

**1. A Golden Test Dataset**
This is a curated set of questions and reference answers that represent the real-world use of your system. Think of it as your "North Star"—the benchmark you optimize against.

**2. Retrieval Metrics**
Ways to measure how well you're retrieving relevant context from your vector store.

**3. Answer Quality Metrics**
Ways to measure how good your final answers are.

With these three components, you can:
- Make a change to your system
- Run evaluations
- See if metrics improved or degraded
- Decide whether to keep the change

This is scientific, quantitative, and repeatable.

## Part 3: Building Your Golden Test Dataset

### What Goes Into a Test Dataset?

For RAG evaluation, each test case typically includes:

**1. Question**: The user's query
```
"Who won the prestigious IoT award in 2023?"
```

**2. Keywords**: Terms that should appear in retrieved chunks
```
["Maxine", "Thompson", "IoT"]
```

**3. Reference Answer**: The ideal response
```
"Maxine Thompson won the prestigious Innovator of the Year (IoTY) award in 2023."
```

**4. Category** (optional but useful): Type of question
```
"direct_fact"
```

### Where to Get Test Data

**Option 1: Create Your Own**
Look through your knowledge base and write questions that your system should be able to answer. This works for new systems but can be biased toward what you think is important.

**Option 2: From Real Users (Best!)**
If you're improving an existing system, you have gold: real user questions and expert-written answers. This is the best source because it reflects actual usage patterns.

For example, if you're automating customer support, you might have:
- Emails from customers (questions)
- Responses from support agents (reference answers)

This is real-world data that perfectly represents your use case.

### Test Dataset Best Practices

**1. Diversity**
Include different types of questions:
- Direct facts: "Who is the CEO?"
- Temporal: "How long has John worked here?"
- Comparative: "Who has been here longer, Alice or Bob?"
- Numerical: "How many employees earn over $60k?"
- Spanning: "List all products with AI features"
- Relationship: "Who does Sarah report to?"
- Holistic: "What are the company's core values?"

**2. Living Dataset**
Your test set should evolve. When you discover a failure in production, add that case to your test set. This ensures you don't regress.

**3. Adequate Size**
In the course example, we use 150 test questions. This is a good starting point. Too few (< 20) and your metrics won't be reliable. Too many (> 1000) and evaluation becomes slow and expensive.

**4. Quality Over Quantity**
It's better to have 50 high-quality, representative test cases than 500 random ones.

### Example Test Dataset Format (JSONL)

```json
{"question": "Who won the prestigious IoT award in 2023?", "keywords": ["Maxine", "Thompson", "IoT"], "reference_answer": "Maxine Thompson won the prestigious Innovator of the Year (IoTY) award in 2023.", "category": "direct_fact"}
{"question": "How long has Avery Lancaster been with the company?", "keywords": ["Avery", "Lancaster", "founded", "2015"], "reference_answer": "Avery Lancaster co-founded InsureElm in 2015, so she has been with the company since its inception.", "category": "temporal"}
```

JSONL (JSON Lines) is a convenient format where each line is a valid JSON object. It's easy to append to and process line by line.

## Part 4: Retrieval Metrics Deep Dive

Retrieval metrics measure how well your vector search is finding relevant chunks. These are "close to the model" metrics—they directly measure what your RAG system is doing.

### Mean Reciprocal Rank (MRR)

**What It Measures:**
Where is the first relevant chunk in your retrieved results?

**How It Works:**
- If the first chunk is relevant: score = 1/1 = 1.0
- If the second chunk is relevant: score = 1/2 = 0.5
- If the third chunk is relevant: score = 1/3 = 0.33
- If the fifth chunk is relevant: score = 1/5 = 0.2

You calculate this for all your test questions and take the average.

**Example:**
```
Test 1: First hit at position 1 → 1.0
Test 2: First hit at position 3 → 0.33
Test 3: First hit at position 2 → 0.5
Test 4: First hit at position 1 → 1.0

MRR = (1.0 + 0.33 + 0.5 + 1.0) / 4 = 0.71
```

**Interpretation:**
- MRR = 1.0: Perfect! Relevant chunk always first
- MRR = 0.5: On average, relevant chunk is 2nd
- MRR = 0.25: On average, relevant chunk is 4th
- MRR < 0.2: Poor retrieval, needs improvement

**Why It's Useful:**
MRR is intuitive and directly relates to user experience. If relevant information is always in the first chunk, your LLM has the best chance of generating a good answer.

### NDCG (Normalized Discounted Cumulative Gain)

**What It Measures:**
How well are ALL your relevant chunks distributed in the retrieved results?

**The Concept:**
Unlike MRR which only looks at the first hit, NDCG considers all relevant chunks. It asks: "Are all the relevant chunks near the top, or are they scattered throughout?"

**How It Works:**
1. For each position, calculate: relevance / log₂(position + 1)
2. Sum these values (Discounted Cumulative Gain)
3. Normalize by the ideal DCG (if all relevant chunks were at the top)

**Why "Discounted"?**
The log₂(position + 1) in the denominator means that relevant chunks further down are worth less. A relevant chunk at position 10 contributes less to the score than one at position 2.

**Example:**
```
Retrieved 5 chunks, relevant ones at positions 1, 2, 5:

DCG = 1/log₂(2) + 1/log₂(3) + 1/log₂(6)
    = 1/1 + 1/1.58 + 1/2.58
    = 1.0 + 0.63 + 0.39
    = 2.02

Ideal DCG (if they were at 1, 2, 3):
IDCG = 1/1 + 1/1.58 + 1/2
     = 1.0 + 0.63 + 0.5
     = 2.13

NDCG = DCG / IDCG = 2.02 / 2.13 = 0.95
```

**Interpretation:**
- NDCG = 1.0: Perfect distribution
- NDCG > 0.8: Good distribution
- NDCG < 0.6: Relevant chunks scattered, poor retrieval

**When to Use:**
NDCG is more sophisticated than MRR and useful when you retrieve many chunks (k > 5) and care about the overall quality of your retrieved set.

### Recall and Precision

These are classic information retrieval metrics.

**Recall: "Did we find what we needed?"**

Formula: (Relevant chunks retrieved) / (Total relevant chunks that exist)

For RAG, we often use "Recall@k":
- Recall@3: In what % of tests did the top 3 chunks contain relevant info?
- Recall@5: In what % of tests did the top 5 chunks contain relevant info?

**Example:**
```
100 test questions, retrieving top 5 chunks each:
- 85 tests had at least one relevant chunk in top 5
- Recall@5 = 85%
```

**Keyword Coverage (Variant of Recall):**
Since we have keywords in our test set, we can measure:
- What % of required keywords appear in retrieved chunks?

```
Question: "Who won the IoT award?"
Keywords: ["Maxine", "Thompson", "IoT"]
Retrieved chunks contain: ["Maxine", "IoT"]
Keyword Coverage = 2/3 = 67%
```

Average this across all tests to get overall keyword coverage.

**Precision: "How much junk did we retrieve?"**

Formula: (Relevant chunks retrieved) / (Total chunks retrieved)

**Example:**
```
Retrieved 5 chunks:
- 3 chunks are relevant
- 2 chunks are irrelevant
Precision = 3/5 = 60%
```

**Why Precision Matters Less for RAG:**
LLMs are pretty good at ignoring irrelevant information in the context. As long as the relevant information is there (recall), a bit of noise (low precision) is usually okay.

However, precision matters if:
- You're trying to minimize context window usage (cost)
- Irrelevant information is actively misleading
- You have very limited context space

### Which Metrics to Use?

**For RAG, prioritize:**
1. **MRR**: Most intuitive, directly impacts answer quality
2. **Keyword Coverage**: Easy to understand, good recall proxy
3. **NDCG**: If you retrieve many chunks and want to understand distribution

**Use precision only if:**
- You're concerned about context pollution
- You're optimizing for cost (fewer tokens)

## Part 5: Answer Quality Metrics

Retrieval metrics tell us if we're finding the right chunks, but what we really care about is: **Are the final answers good?**

### The Challenge of Evaluating Answers

How do you automatically evaluate if an answer is good? 

**Bad Approach: Exact String Matching**
```python
if generated_answer == reference_answer:
    score = 1.0
else:
    score = 0.0
```

This is too strict. The answer "Maxine Thompson won the award" is just as good as "The award was won by Maxine Thompson", but they're not exact matches.

**Better But Still Janky: Word Overlap**
Count how many words from the reference answer appear in the generated answer. Better, but doesn't understand semantics.

**Best Approach: LLM-as-a-Judge**

Use another LLM to evaluate the answer! LLMs are excellent at:
- Understanding semantic similarity
- Judging quality on multiple dimensions
- Providing consistent scores

### LLM-as-a-Judge Implementation

**The Prompt:**
```
You are an expert evaluator assessing the quality of answers.

Question: {question}
Reference Answer: {reference_answer}
Generated Answer: {generated_answer}

Evaluate the generated answer on these dimensions:
1. Accuracy: Is the information correct?
2. Completeness: Does it include all necessary information?
3. Relevance: Is all information relevant to the question?

Provide scores from 1-5 for each dimension.
```

**Example Evaluation:**
```
Question: "Who won the IoT award?"
Reference: "Maxine Thompson won the IoT award in 2023."
Generated: "Maxine won the award."

Scores:
- Accuracy: 5/5 (Maxine is correct)
- Completeness: 4/5 (Missing last name Thompson)
- Relevance: 5/5 (All information is relevant)
```

### The Three Dimensions Explained

**1. Accuracy**
Is the information factually correct?

- 5/5: All facts are correct
- 4/5: Minor inaccuracies
- 3/5: Some significant errors
- 2/5: Major errors
- 1/5: Completely wrong

**Example:**
- Question: "What year was the company founded?"
- Reference: "2015"
- Generated: "2015" → Accuracy: 5/5
- Generated: "2016" → Accuracy: 1/5

**2. Completeness**
Does the answer include all necessary information?

- 5/5: Complete answer
- 4/5: Missing minor details
- 3/5: Missing important information
- 2/5: Very incomplete
- 1/5: Almost no information

**Example:**
- Question: "Who won the IoT award?"
- Reference: "Maxine Thompson"
- Generated: "Maxine Thompson" → Completeness: 5/5
- Generated: "Maxine" → Completeness: 4/5 (missing last name)
- Generated: "An employee" → Completeness: 2/5

**3. Relevance**
Is there unnecessary or off-topic information?

- 5/5: All information is relevant
- 4/5: Minor irrelevant details
- 3/5: Some off-topic content
- 2/5: Mostly irrelevant
- 1/5: Completely off-topic

**Example:**
- Question: "Who won the IoT award?"
- Generated: "Maxine Thompson" → Relevance: 5/5
- Generated: "Maxine Thompson, who joined in 2018, won the award" → Relevance: 4/5 (join date not asked for)
- Generated: "Maxine Thompson won the award. She likes coffee." → Relevance: 3/5

### Best Practices for LLM-as-a-Judge

**1. Use a Strong Model**
The judge should be at least as capable as the model generating answers. In the course, we use GPT-4o-mini or GPT-4o-nano, which are strong enough for this task.

**2. Structured Outputs**
Force the LLM to return scores in a consistent format using structured outputs (we'll see this in the code). This makes parsing and aggregation easy.

**3. Be Strict**
In your prompt, tell the judge: "Only give 5/5 for perfect answers." This prevents grade inflation.

**4. Provide Examples (Optional)**
You can include few-shot examples in your prompt to calibrate the judge's scoring.

### Cost Considerations

LLM-as-a-Judge requires an API call for each test question. With 150 tests:
- Using GPT-4o-mini: ~$0.02-0.03
- Using GPT-4o-nano: ~$0.01-0.02
- Using open-source (local): Free!

This is very affordable and worth it for the quality of evaluation.

## Part 6: Running Experiments

Now that we have our evaluation framework, let's see it in action!

### Baseline Performance (Day 15 System)

Our starting point with basic RAG:
- Chunk size: 1000 characters
- Encoder: all-MiniLM-L6-v2 (HuggingFace)
- Retrieval: Top 5 chunks
- No advanced techniques

**Results:**
- MRR: 0.7298 🔴
- NDCG: 0.7387
- Keyword Coverage: 83.8%
- Accuracy: 3.99 🔴
- Completeness: 3.85 🔴
- Relevance: 4.57 🟡

**Interpretation:**
- Relevant chunks typically in 2nd-3rd position (not ideal)
- Missing ~16% of keywords
- Answers are incomplete and sometimes inaccurate
- At least we're not adding irrelevant information

### Experiment 1: Chunk Size Optimization

**Hypothesis:** Smaller chunks might match queries better.

**Test 1: Chunk Size = 500**
- Retrieve top 10 chunks (to keep total context similar)
- Results: MRR = 0.7604 ✅

**Test 2: Chunk Size = 1667**
- Retrieve top 3 chunks
- Results: MRR = 0.7475

**Test 3: Markdown Splitter**
- Let LangChain split by markdown headers
- Results: MRR = 0.738

**Winner:** 500-character chunks with top-10 retrieval

**Why This Works:**
Smaller chunks are more focused. A query about "IoT award" is more likely to match a small chunk specifically about the award than a large chunk about someone's entire career.

**The Trade-off:**
Too small and you lose context. If chunks are 50 characters, you might get "won the award" without knowing who "she" refers to. Balance is key.

### Experiment 2: Encoder Models

**Hypothesis:** Better embedding models → better retrieval.

**Test 1: OpenAI text-embedding-3-small**
- 1536 dimensions
- Results: MRR = 0.7849 ✅

**Test 2: OpenAI text-embedding-3-large**
- 3072 dimensions
- Results: MRR = 0.7903 ✅✅

**Winner:** text-embedding-3-large

**Why This Works:**
OpenAI's embedding models are trained on massive, diverse datasets and are specifically optimized for semantic search. They capture nuances that the smaller HuggingFace model misses.

**Cost Consideration:**
- HuggingFace: Free (local)
- OpenAI small: ~$0.02 per 1M tokens
- OpenAI large: ~$0.13 per 1M tokens

For 970 chunks (~500 chars each), that's ~$0.06 with the large model. Very affordable!

### Combined Results After Basic Optimization

After optimizing chunk size and encoder:
- MRR: 0.7903 🟡 (up from 0.7298)
- Accuracy: 4.21 🟡 (up from 3.99)

We've moved from RED to YELLOW territory. Good progress, but we can do better!

## Part 7: The Zoo of Advanced RAG Techniques

Now we enter the world of advanced RAG. There are many techniques, each with fancy names. Let me demystify them.

### Important Philosophy

Before we dive in, remember:
- **There is no "best" technique**
- **Your mileage will vary**
- **Always measure, never assume**

These techniques are tools in your toolbox. Use them when they solve specific problems you've identified through evaluation.

### Technique 1: Chunking R&D

**What:** Experiment with different chunking strategies.

**Options:**
- RecursiveCharacterTextSplitter (what we used)
- MarkdownHeaderTextSplitter (splits on headers)
- Semantic Chunking (LLM-based, we'll implement this!)
- Custom splitters for your domain

**When to Use:**
- You notice chunks split important information
- Questions fail because context is fragmented
- You have structured documents (markdown, HTML, etc.)

**Example Problem:**
"Who won the IoT award?" fails because "Maxine" is in one chunk and "Thompson" is in another.

**Solution:**
Use semantic chunking to keep related information together.

### Technique 2: Encoder Selection

**What:** Try different embedding models.

**Options:**
- HuggingFace models (free, local)
  - all-MiniLM-L6-v2 (384 dim)
  - all-mpnet-base-v2 (768 dim)
- OpenAI models (paid, API)
  - text-embedding-3-small (1536 dim)
  - text-embedding-3-large (3072 dim)
- Domain-specific models (medical, legal, code)

**When to Use:**
- Low retrieval metrics
- Domain-specific vocabulary
- Multilingual requirements

**Special Case: Images**
If your knowledge base includes images:
1. Use vision model to generate captions
2. Embed the captions (easier than multi-modal embeddings)
3. Store image path in metadata

### Technique 3: Prompt Engineering

**What:** Improve the prompts you use for answer generation.

**Often Overlooked!**
People get excited about fancy RAG techniques and forget that simple prompt improvements can have huge impact.

**What to Add:**
- Current date/time
- Company/domain context
- Explicit instructions about evaluation criteria
- Conversation history
- Output format requirements

**Example:**
```python
# Before
system_prompt = "You are a helpful assistant."

# After
system_prompt = """You are a knowledgeable assistant for InsureElm.
Today's date: {current_date}

Your answers will be evaluated for:
- Accuracy: Is the information correct?
- Completeness: Does it fully answer the question?
- Relevance: Is all information relevant?

Be concise but complete. Use the provided context."""
```

This simple change can boost accuracy by 0.2-0.3 points!

### Technique 4: Document Pre-processing

**What:** Rewrite documents before embedding them.

**The Problem:**
Some documents aren't naturally suited for retrieval. Examples:
- Tables of numbers
- Bullet-point lists
- Technical specifications
- Code snippets

**The Solution:**
Use an LLM to rewrite them into natural language that's easier to match with queries.

**Example:**
```
Original document:
| Route | Price |
|-------|-------|
| NYC-LA | $350 |
| NYC-CHI | $150 |

Rewritten:
"Ticket prices: New York to Los Angeles costs $350. 
New York to Chicago costs $150."
```

Now a query like "How much to fly to LA?" will match better!

**Semantic Chunking (Advanced):**
Don't just split mechanically—use an LLM to:
1. Identify meaningful sections
2. Create a headline for each section
3. Write a summary
4. Keep the original text

This is what we'll implement in our advanced version!

### Technique 5: Query Rewriting

**What:** Rewrite the user's question before retrieval.

**The Problem:**
Users don't always ask questions in the optimal way for retrieval.

**Examples:**
- Follow-up question: "What about his salary?" (who is "his"?)
- Ambiguous: "Tell me about the award" (which award?)
- Verbose: "I was wondering if you could possibly tell me..."

**The Solution:**
Use an LLM to rewrite the query:

```python
def rewrite_query(question, history):
    prompt = f"""Given this conversation history:
    {history}
    
    User's question: {question}
    
    Rewrite as a clear, standalone search query."""
    
    return llm.complete(prompt)
```

**Example:**
```
History: "Tell me about Maxine Thompson"
Question: "What award did she win?"

Rewritten: "What award did Maxine Thompson win?"
```

**Caveat:**
Query rewriting can sometimes hurt! It might add words that dilute the search (like adding "InsureElm" to every query). We'll handle this with query expansion.

### Technique 6: Query Expansion

**What:** Generate multiple query variations and retrieve for each.

**The Idea:**
Instead of one query, create 2-3 variations and retrieve chunks for all of them.

**Example:**
```
Original: "Who won the IoT award?"

Expanded:
1. "Who won the IoT award?"
2. "IoT award recipient"
3. "Innovator of the Year award winner"
```

Retrieve 5 chunks for each query → 15 total chunks → merge and deduplicate → rerank.

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
4. Chunk about Manchester office location

After reranking:
1. Chunk about Jessica Liu's education ← moved to top!
2. Chunk about university partnerships
3. Chunk about Manchester office location
4. Chunk about London universities
```

**Cost:**
One LLM call per question. With GPT-4o-nano, ~$0.001 per reranking. Very affordable!

### Technique 8: Hierarchical RAG

**What:** Create summaries at different levels of granularity.

**The Problem RAG Struggles With:**
"How many employees earn over $60k?"

This requires information from many documents. RAG typically retrieves 5-10 chunks, but you might need 50+ employee records to answer this.

**The Solution:**
Create summary documents:
- All employees → salary summary (one document)
- All products → feature comparison (one document)
- All contracts → terms summary (one document)

When you query, first search summaries, then drill down if needed.

**Example:**
```
Summary document: "employee_salaries_summary.md"
"Employee salary ranges: 15 employees earn $40k-60k, 
23 employees earn $60k-80k, 8 employees earn $80k+."

Query: "How many employees earn over $60k?"
Retrieved: salary summary
Answer: "31 employees (23 + 8)"
```

**Caveat:**
You need to anticipate what summaries will be useful. It's still a bit of a hack—you can't predict every possible spanning question.

### Technique 9: Graph RAG

**What:** Store and leverage relationships between documents.

**When Documents Are Connected:**
- Employee → Manager (another employee)
- Product → Related products
- Document → Source documents
- Code file → Imported modules

**Approach 1: Metadata**
```python
chunk_metadata = {
    "source": "employee_maxine.md",
    "employee_id": "E123",
    "manager_id": "E001",  # Avery Lancaster
    "department": "Engineering"
}
```

When you retrieve Maxine's chunk, also retrieve her manager's chunk.

**Approach 2: Graph Database**
Use Neo4j or similar to store documents as nodes with relationships as edges. When you retrieve a node, also get neighbors (1-2 hops away).

**When to Use:**
- Highly connected data
- Relationship queries ("Who reports to whom?")
- Network analysis ("Find all employees connected to this project")

**Reality Check:**
Graph RAG is trendy but often overkill. Metadata relationships work for most cases. Only use a full graph database if you have complex, multi-hop relationship queries.

### Technique 10: Agentic RAG

**What:** Let an LLM decide how to retrieve information.

**Traditional RAG (Hardcoded):**
```
User question → Vectorize → Vector search → Top-k chunks → LLM answer
```

**Agentic RAG (LLM-Orchestrated):**
```
User question → LLM with tools:
  - Tool 1: Vector search
  - Tool 2: Keyword search
  - Tool 3: SQL query
  - Tool 4: API call

LLM decides which tools to use, in what order, and how many times.
```

**Example:**
```
Question: "How many employees in Engineering earn over $60k?"

Agent's thought process:
1. Use SQL tool: "SELECT COUNT(*) FROM employees WHERE department='Engineering' AND salary > 60000"
2. Result: 12 employees
3. Use vector search: "Engineering department employees"
4. Combine information
5. Generate answer: "There are 12 employees in Engineering earning over $60k..."
```

**Advantages:**
- Flexible (handles unexpected questions)
- Can iterate (if first approach fails, try another)
- Powerful for complex queries

**Disadvantages:**
- Less predictable
- Higher cost (multiple LLM calls)
- Slower
- Harder to debug

**When to Use:**
- Complex, multi-step questions
- Unpredictable query types
- When you can afford the latency and cost

**Is This Still RAG?**
Yes! It's still retrieving information to augment generation. Just with LLM orchestration instead of hardcoded logic.

## Part 8: Advanced Implementation (No LangChain)

Now let's get our hands dirty with code! We're going to rebuild our RAG system without LangChain to:
1. Understand what's really happening
2. Have full control
3. Implement advanced techniques

### Why No LangChain?

LangChain is great for rapid prototyping, but:
- Abstractions hide what's happening
- Less control over each step
- Framework lock-in

By going native, you'll understand the fundamentals and be able to use any framework (or none) in the future.

### Key Components

**1. Direct Chroma Usage**
```python
from chromadb import PersistentClient

# Create client
client = PersistentClient(path="./vector_db")

# Create collection
collection = client.create_collection("my_docs")

# Add documents
collection.add(
    embeddings=vectors,
    documents=texts,
    metadatas=metadata_list,
    ids=ids
)

# Query
results = collection.query(
    query_embeddings=query_vector,
    n_results=5
)
```

No LangChain wrapper—direct API calls!

**2. OpenAI Embeddings API**
```python
import openai

response = openai.embeddings.create(
    model="text-embedding-3-large",
    input=texts
)

embeddings = [item.embedding for item in response.data]
```

Clear separation: we call OpenAI for embeddings, then Chroma for storage.

**3. Custom Document Class**
```python
from pydantic import BaseModel

class Result:
    """Like LangChain's Document, but ours"""
    page_content: str
    metadata: dict
```

Simple and clear!

### Semantic Chunking with LLMs

This is where it gets exciting. Instead of mechanically splitting at character counts, we'll use an LLM to intelligently chunk documents.

**The Process:**
```
For each document:
1. Send to LLM with prompt: "Divide this into meaningful chunks"
2. LLM analyzes content and identifies logical sections
3. For each section, LLM creates:
   - Headline (brief title)
   - Summary (2-3 sentences)
   - Original text (preserved)
4. Store all three in the chunk
```

**Why This Works:**
- Chunks align with semantic meaning
- Headlines and summaries improve retrieval
- Original text preserved for answer generation

**The Prompt:**
```python
prompt = f"""You are a document chunking expert.

Task: Split this document into overlapping chunks for a knowledge base.

Document type: {doc_type}
Source: {source}

Guidelines:
- Divide into approximately {num_chunks} chunks
- Each chunk should be a meaningful section
- Chunks can overlap slightly
- For each chunk, provide:
  * headline: Brief title (few words)
  * summary: 2-3 sentence summary
  * original_text: The actual text from the document

Document:
{document_text}

Respond with the chunks."""
```

**Structured Outputs:**
```python
from pydantic import BaseModel

class Chunk(BaseModel):
    headline: str
    summary: str
    original_text: str

class Chunks(BaseModel):
    chunks: list[Chunk]

# Force LLM to return this structure
response = llm.complete(
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
```python
prompt = f"""You are in conversation with a user.

Conversation history:
{history}

User's question: {question}

Respond with a single refined question that will be used 
to search the knowledge base. 

Important: Respond ONLY with the query, nothing else."""
```

**Example:**
```
History:
User: "Tell me about Maxine Thompson"
Assistant: "Maxine is a Senior Data Engineer..."

New question: "What award did she win?"

Rewritten: "What award did Maxine Thompson win?"
```

Now the query is self-contained and will retrieve better!

### Query Expansion Strategy

**The Challenge:**
Query rewriting sometimes helps, sometimes hurts. It might add noise like "InsureElm" to every query.

**Our Solution:**
Don't replace the original query—use BOTH!

```python
def fetch_context(question, history):
    # Get rewritten query
    rewritten = rewrite_query(question, history)
    
    # Retrieve for both
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
    chunks2 = retrieve(rewritten, k=20)
    merged = merge_unique(chunks1, chunks2)
    
    # 3. Reranking
    reranked = llm_rerank(question, merged)
    top_chunks = reranked[:10]
    
    # 4. Generate answer
    context = "\n\n".join([c.page_content for c in top_chunks])
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    answer = llm.complete(messages)
    return answer
```

### The Complete Flow

```
User asks question
    ↓
Query rewriting (LLM call 1)
    ↓
Query expansion (retrieve for original + rewritten)
    ↓
Merge chunks (remove duplicates)
    ↓
Reranking (LLM call 2)
    ↓
Keep top 10 chunks
    ↓
Generate answer (LLM call 3)
    ↓
Return answer to user
```

Total: 3 LLM calls per question (fast and affordable!)

## Part 10: Results and Analysis

### Final Performance

After implementing all advanced techniques:

**Retrieval Metrics:**
- MRR: 0.9116 🟢 (was 0.7298)
- NDCG: 0.9025 🟢 (was 0.7387)
- Keyword Coverage: 96% 🟢 (was 83.8%)

**Answer Quality:**
- Accuracy: 4.62 🟢 (was 3.99)
- Completeness: 4.35 🟡 (was 3.85)
- Relevance: 4.84 🟢 (was 4.57)

### What Made the Difference?

**1. Semantic Chunking (+0.05 MRR)**
LLM-based chunking kept related information together. "Maxine Thompson" no longer split across chunks.

**2. Better Encoder (+0.06 MRR)**
text-embedding-3-large captures semantic nuances better than all-MiniLM-L6-v2.

**3. Query Expansion (+0.04 MRR)**
Using both original and rewritten queries increased recall without sacrificing precision.

**4. Reranking (+0.08 MRR)**
LLM reranking moved the most relevant chunks to the top, dramatically improving answer quality.

**5. Improved Prompts (+0.3 Accuracy)**
Explicitly telling the LLM it will be evaluated on accuracy, completeness, and relevance made it try harder!

### Cost Analysis

**One-time Ingest Cost:**
- 76 documents × semantic chunking = ~$0.30
- 534 chunks × embeddings (large) = ~$0.05
- Total: ~$0.35

**Per-Query Cost:**
- Query rewriting: $0.0001
- Reranking: $0.0003
- Answer generation: $0.0005
- Total: ~$0.001 per question

**For 150 test evaluations:**
- Retrieval eval: Free (just vector search)
- Answer eval (LLM judge): $0.02
- Total: ~$0.02

**Very affordable for production!**

### Performance by Question Category

```
Category          | MRR   | Accuracy
------------------|-------|----------
Direct Facts      | 0.95  | 4.8
Temporal          | 0.89  | 4.5
Comparative       | 0.87  | 4.4
Numerical         | 0.91  | 4.3
Relationship      | 0.88  | 4.5
Spanning          | 0.82  | 4.0
Holistic          | 0.79  | 3.9
```

**Observations:**
- Direct facts: Excellent (as expected)
- Spanning/Holistic: Still challenging (RAG's Achilles heel)
- Numerical: Good (better than expected)

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

People get excited about fancy techniques (Graph RAG! Agentic RAG!) and forget that improving prompts can have huge impact with zero additional cost.

**Example:**
Adding "Your answer will be evaluated for accuracy, completeness, and relevance" boosted our accuracy score by 0.3 points!

### 4. Focus on Business Impact

**Retrieval metrics guide optimization, but answer quality matters most.**

High MRR is great, but if your answers are still wrong, users won't be happy.

**Balance:**
- Use retrieval metrics for fast iteration
- Use answer metrics to validate business value
- Collect real user feedback when possible

### 5. Know When to Stop

**Diminishing returns are real.**

Going from 0.73 to 0.79 MRR is relatively easy. Going from 0.91 to 0.95 might require weeks of work.

**Ask yourself:**
- Is 0.91 MRR good enough for my use case?
- Will users notice the difference between 4.6 and 4.8 accuracy?
- Is my time better spent on other features?

### 6. Test with Real Users

**Your test set is a proxy, not reality.**

Even with 150 carefully curated test questions, you might miss edge cases. Deploy to a small group of real users and gather feedback.

**Strategies:**
- Beta testing with internal users
- Thumbs up/down on answers
- "Was this helpful?" feedback
- Monitor questions that return "I don't know"

### 7. Keep Your Test Set Updated

**Your test set should evolve.**

When you discover a failure in production, add it to your test set. This ensures you don't regress.

**Example:**
User asks "Who went to Manchester University?" → System fails → Add to test set → Fix the issue → Now it's tested forever

## Part 12: Common Pitfalls and How to Avoid Them

### Pitfall 1: Over-Optimizing Retrieval

**The Trap:**
You achieve MRR = 0.95, keyword coverage = 98%, but answer accuracy is still 4.0.

**Why This Happens:**
Retrieval metrics don't guarantee good answers. The LLM might still misinterpret the context or hallucinate.

**Solution:**
Balance retrieval and answer metrics. If retrieval is good but answers are bad, focus on:
- Prompt engineering
- Using a better LLM for answer generation
- Providing more context in the prompt

### Pitfall 2: Premature Complexity

**The Trap:**
You immediately implement Graph RAG, Agentic RAG, and Hierarchical RAG before even testing basic RAG.

**Why This Happens:**
Advanced techniques sound cool! But they add complexity, cost, and latency.

**Solution:**
Start simple:
1. Basic RAG
2. Measure
3. Identify specific failures
4. Add techniques to fix those failures
5. Measure again

Only add complexity when you have evidence it will help.

### Pitfall 3: Ignoring Cost

**The Trap:**
You implement query expansion with 5 queries, reranking with GPT-4, and answer generation with GPT-4. Cost per question: $0.05. With 10,000 questions/day, that's $500/day!

**Why This Happens:**
During development, you're not thinking about scale.

**Solution:**
- Use cheaper models where possible (GPT-4o-mini for reranking)
- Cache common queries
- Implement rate limiting
- Monitor costs in production

### Pitfall 4: Forgetting About Latency

**The Trap:**
Your RAG pipeline makes 5 LLM calls per question. Each takes 2 seconds. Total: 10 seconds per answer. Users leave before seeing results.

**Why This Happens:**
You optimize for quality without considering user experience.

**Solution:**
- Parallelize where possible (query expansion can run in parallel)
- Use streaming for answer generation (show partial results)
- Set latency budgets (e.g., "answers must be < 5 seconds")
- Consider async processing for non-urgent queries

### Pitfall 5: Not Handling Failures Gracefully

**The Trap:**
Your system crashes when:
- No chunks are retrieved
- LLM returns malformed JSON
- API rate limit is hit

**Why This Happens:**
You test the happy path but not edge cases.

**Solution:**
```python
try:
    chunks = retrieve(question)
    if not chunks:
        return "I don't have information about that."
    
    answer = llm.complete(prompt)
    return answer
except RateLimitError:
    return "I'm experiencing high load. Please try again."
except Exception as e:
    log_error(e)
    return "I encountered an error. Please try rephrasing your question."
```

Always have fallbacks!

## Part 13: The Path Forward

### Immediate Next Steps

**1. Beat the Benchmark**
Take the provided code and try to improve the metrics:
- Target: MRR > 0.92, Accuracy > 4.7
- Document your experiments
- Share your results!

**2. Apply to Your Domain**
Build a RAG system for:
- Your company's documentation
- Your personal notes and files
- A specific domain (medical, legal, etc.)

**3. Implement One Advanced Technique**
Pick one we didn't fully implement:
- Hierarchical RAG
- Graph RAG
- Agentic RAG

### The Agentic Challenge (Advanced)

**Level 1: Simple Tool**
Add a keyword search tool to your RAG system:
```python
def keyword_search(keyword: str) -> list[str]:
    """Search all documents for exact keyword match"""
    # Return documents containing the keyword
```

**Level 2: Full Agentic Retrieval**
Give the LLM multiple tools:
- Vector search
- Keyword search
- SQL query (if you have structured data)
- File listing

Let it decide how to retrieve information.

**Level 3: Self-Evaluating Agent**
The ultimate challenge:
```python
def self_evaluating_rag(question):
    max_attempts = 5
    for attempt in range(max_attempts):
        answer = generate_answer(question)
        evaluation = llm_judge(question, answer)
        
        if evaluation.accuracy == 5 and evaluation.completeness == 5:
            return answer
        
        # Feedback loop
        feedback = f"Your answer scored {evaluation.accuracy}/5 on accuracy and {evaluation.completeness}/5 on completeness. Try again with more focus on {evaluation.feedback}"
        
    return answer  # Return best attempt
```

This could theoretically achieve 5/5 on all metrics!

### Beyond RAG

**Week 6-8 Preview:**
- Data science foundations
- Fine-tuning LLMs
- Production deployment
- Capstone project

**Don't stop after RAG!**
The next weeks cover different but equally important topics. You'll become a well-rounded LLM engineer.

## Conclusion

Today you learned:

**1. The Importance of Evaluation**
- Build golden test datasets
- Measure retrieval (MRR, NDCG, recall)
- Measure answers (LLM-as-a-judge)
- Iterate scientifically

**2. Advanced RAG Techniques**
- Semantic chunking
- Query rewriting and expansion
- Reranking
- And 7 more techniques!

**3. Production Implementation**
- Native Chroma usage
- Structured outputs
- Multiprocessing
- Cost and latency optimization

**4. Real Results**
- MRR: 0.73 → 0.91 (25% improvement)
- Accuracy: 3.99 → 4.62 (16% improvement)
- Production-ready performance

**Most Importantly:**
You now have a framework for building, evaluating, and optimizing RAG systems for any domain.

**Remember:**
- RAG is empirical—always measure
- Start simple, add complexity only when needed
- Focus on business impact, not just metrics
- Keep iterating!

**Now go build something amazing! 🚀**

---

## Additional Resources

### Papers and Articles
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (original RAG paper)
- "Evaluating Retrieval Augmented Generation: A Survey"
- "Lost in the Middle: How Language Models Use Long Contexts"

### Tools and Libraries
- **Chroma**: https://www.trychroma.com/
- **LiteLLM**: https://litellm.ai/
- **LangChain** (optional): https://python.langchain.com/
- **Pydantic**: https://docs.pydantic.dev/

### Embedding Model Benchmarks
- **MTEB Leaderboard**: https://huggingface.co/spaces/mteb/leaderboard
- Compare embedding models for your use case

### Community
- Share your results on LinkedIn
- Join RAG-focused Discord communities
- Contribute to open-source RAG projects

### Further Learning
- Week 6: Data Science & Fine-tuning
- Week 7-8: Capstone Project
- Agentic AI Course (for Agentic RAG)

---

**Congratulations on completing Day 16! You're now a RAG expert! 🎉**

