# Day 14 Lecture Notes: Model Selection and Introduction to RAG

## Introduction

Welcome to Day 14 of our LLM Engineering course! We're now in Day 14, and I have some exciting news. We only have one more week until RAG week, which is where everyone wants to be. This week is going to be relatively short, quick, and sharp compared to the previous weeks. Don't skip this week though, because it's important foundation for what's coming.

## Course Progress Review

Let's quickly review where we are in the course:

**Day 11-12: Foundations**
- Chat Completions API basics
- Frontend models and APIs
- Tool calling capabilities
- Multi-modality features

**Day 13: Hugging Face Deep Dive**
- Transformers library
- Pipelines for quick implementations
- Working with tokenizers and models directly

**Day 14 (Current): Model Selection**
- Selecting the right models
- Understanding benchmarks
- Generating code with LLMs

**Day 14-15 (Next): RAG Week**
- Retrieval Augmented Generation
- Building production RAG systems

By the end of this course, you'll be able to confidently say you've mastered LLM engineering!

## The Central Question: Which is the Best LLM?

This is the question everyone asks, and it's actually an ill-posed question. There is no single "best" LLM. The right question to ask is:

**"What's the right model for the task at hand, given the problem I'm trying to solve?"**

### Strategy for Selecting Models

The process of selecting the right model involves several steps:

1. **Understand your requirements** - What are you trying to build?
2. **Look at basic model information** - Parameters, cost, context window, etc.
3. **Examine benchmarks** - How does it perform on relevant tasks?
4. **Test with your specific use case** - Benchmarks only take you so far

## Understanding Model Basics

When evaluating models, you need to look at several fundamental characteristics:

### 1. Model Type

**Open Source vs. Closed Source**
- Open source models can be run locally or on your infrastructure
- Closed source (paid) models are accessed via APIs
- Each has trade-offs in cost, control, and performance

**Chat vs. Reasoning vs. Hybrid**
- **Chat models**: Fast, creative, good for general conversation
- **Reasoning models**: Slower, better at complex problem-solving, "think" before responding
- **Hybrid models**: Can choose between modes based on the situation

It's not always the case that a reasoning model is better. Reasoning models:
- Often perform better on benchmarks
- Can think through complex problems
- Tend to be slower
- May not be as good at generating creative content

### 2. Temporal Information

**Release Date**
- How recent is the model?
- Newer models generally incorporate better techniques

**Knowledge Cutoff**
- When was the model's training data collected up to?
- If trained until July 2024, it knows facts up to that point
- For more recent information, you need inference-time techniques (like RAG!)

### 3. Model Capacity

**Number of Parameters**
- Indicates the model's capacity to absorb information
- Generally, bigger is better (more parameters = smarter)
- However, recent advances allow smaller models to be surprisingly powerful
- Examples: Llama 3.2 (small but powerful), Gemma 3 (270M parameters but impressive for its size)

**Training Tokens**
- How much data was used to train the model
- More training data generally means more knowledge
- Must be balanced with parameter count (see Chinchilla Scaling Law)

### 4. Context Window

The context window is the total capacity the model has to look back on information from the conversation. This includes:
- The first message you sent
- The model's reply
- The next message
- All subsequent messages
- The current response being generated

Everything must fit within the context window. Larger context windows allow for:
- Longer conversations
- More context in prompts
- Bigger documents to be processed at once

### 5. Finding This Information

You can find basic model information in several places:
- **Model Cards**: Published by providers like OpenAI, Anthropic, Google
- **Leaderboards**: Compare multiple models side-by-side
- **Provider Documentation**: Official specs and pricing

## Additional Model Considerations

Beyond the basics, there are several other factors that significantly impact model selection:

### Cost Analysis

**API Costs**
- For cloud-based frontier models
- Usually priced per token (input and output separately)
- Can vary significantly between models

**Compute Costs**
- For locally-run models
- Even "free" local models use your hardware resources
- Consider server costs if running on company infrastructure

**Training Costs**
- If you plan to fine-tune the model
- Can be substantial for larger models

**Build Costs**
- Development time and resources
- Testing and iteration

### Time to Market

This is closely related to cost and model choice:
- Using GPT-5 or Claude 4.5: Quick to market, less build complexity
- Using smaller/cheaper models: Lower runtime costs, but more time needed for:
  - Experimentation
  - Optimization
  - Getting desired performance

**Trade-off**: Larger, more expensive models = faster development but higher ongoing costs

### Performance Characteristics

**Speed (Throughput)**
- How fast does the model respond?
- Measured in tokens per second
- Examples:
  - Gemini Flash Light: Blazingly fast
  - GPT-4o Nano: Very fast
  - Larger reasoning models: Slower

**Latency (Time to First Token)**
- How long until the model starts responding?
- Particularly important for reasoning models that "think" first
- Critical for user experience in interactive applications

**Rate Limits**
- How many requests can you make?
- Often tied to pricing tiers
- Can usually pay more for higher limits
- Even top tiers have some limits (though enterprise deals may exist)

### Licensing

Often overlooked but critically important, especially for open source models:

**License Types**
- **Permissive**: Use for any purpose
- **Commercial Restrictions**: Revenue caps or usage limitations
- **Special Agreements**: Like Meta's models requiring signed agreements

**Why It Matters**
- Legal compliance
- Commercial viability
- Competitive restrictions

Always involve legal counsel when selecting models for commercial applications.

## The Chinchilla Scaling Law

The Chinchilla Scaling Law was published by Google (named after their model "Chinchilla," which followed "Gopher"). While it's emphasized less now, it's still a valuable rule of thumb.

### The Core Principle

**The relationship between parameters and training data should be proportional.**

If you double the number of parameters in your model, you need approximately twice as much training data to take full advantage of those extra parameters.

### How to Think About It

Imagine you're training a model with 8 billion parameters:
- You're halfway through your training data
- The model stops improving (diminishing returns)
- This signals you need more parameters to absorb the remaining data

Conversely:
- You have a model and want to double its parameters
- You'll need twice as much training data to get the benefit
- Without enough data, extra parameters won't help

### Why It's Less Emphasized Now

Two main reasons:

1. **Better Compression Techniques**
   - Improved training methods
   - Model pruning (removing unnecessary weights)
   - Better architectures
   - Result: Smaller models can hold more information
   - Example: Llama 3.2 is much smaller but still powerful

2. **Inference-Time Techniques**
   - Performance gains now come from what you do at inference time
   - Reasoning capabilities
   - RAG (coming soon!)
   - Tool use
   - These techniques matter as much as training now

### The Takeaway

Given the same model architecture and training techniques:
- 2x parameters → need 2x training tokens
- But modern techniques can break this rule through better compression and inference-time improvements

## Understanding Benchmarks

After reviewing basics to create a shortlist, the next step is examining benchmarks. Benchmarks provide early guidance but aren't the final validation method (we'll cover complete validation strategies later).

### Why Hard Benchmarks Matter

There are many ways to assess models, but a few benchmarks are particularly challenging and help differentiate top models. Let's explore six important ones:

### 1. GPQA (Google-Proof Q&A)

**What It Is**
- 448 difficult questions in physics, chemistry, and biology
- Designed to be "Google-proof"

**Performance Levels**
- **Non-PhD humans with Google access (30 minutes)**: 34%
- **PhD-level humans**: 65%
- **GPT-4 (when benchmark released)**: 39%
- **Current top models**: ~88%

**Why "Google-Proof"?**
Even with access to Google and 30 minutes of research, non-experts only score 34%. These questions require deep understanding, not just information lookup.

**Source**: https://arxiv.org/abs/2311.12022

### 2. MMLU Pro

**Background**
- Original MMLU (Massive Multitask Language Understanding) became too easy
- Had ambiguity issues
- Models were saturating at 99%

**MMLU Pro Improvements**
- 10 answer choices instead of 4 (harder)
- Ambiguity removed
- More difficult questions
- Robust and trustworthy metric

**What It Tests**
- Broad language understanding across many domains
- Ability to distinguish between many possible answers

### 3. AIME (American Invitational Mathematics Examination)

**What It Is**
- Competitive mathematics competition for high school students
- Not your typical high school test!
- For top-performing students who compete at Math Olympiad level

**What It Tests**
- Mathematical problem-solving
- Not mental arithmetic
- Complex math puzzles
- Requires creative thinking and deep understanding

**Note**: These are genuinely difficult problems that most adults (including the instructor) couldn't solve.

### 4. Live Code Bench

**What It Is**
- Benchmark for evaluating coding ability
- Problems from competitive coding sites (LeetCode, Codeforces)

**Special Feature**
- Constantly updated with new problems
- Prevents training data contamination
- Models can't just memorize solutions

**What It Tests**
- Practical coding skills
- Problem-solving in code
- Algorithm implementation

**Source**: https://livecodebench.github.io/

### 5. MUSA (Multi-step Soft Reasoning)

**What It Is**
- Tests reasoning through complex scenarios
- Particularly designed to highlight reasoning model capabilities

**Categories**
The most interesting category involves murder mysteries:
- Given a 1,000-word murder mystery
- Model must identify who has:
  - The means
  - The motive
  - The opportunity

**Why It's Great**
- Real-world puzzle format
- Requires multi-step reasoning
- Tests comprehension and logical deduction
- Tangible, relatable task

### 6. HLE (Humanity's Last Exam)

**The Concept**
- Designed to test superhuman intelligence
- Meant to be the hardest test we can create
- Response to models quickly mastering previous "hard" benchmarks

**Specifications**
- 2,500 insanely complex questions
- Questions that seem impossible to most humans

**Performance Trajectory**
- Late 2024 (release): Models scored 2-3%
- Current top models: 26.5%
- Huge progress in just months

**Why It Exists**
Every time we create a "hard" test, models master it within a year. HLE is an attempt to create a benchmark that will remain challenging longer.

**Source**: https://huggingface.co/spaces/HumanityLastExam/leaderboard

## Benchmark Limitations

While benchmarks are useful indicators, they have serious limitations. Use them as a first indication, but take them with a heavy pinch of salt.

### 1. Training Data Contamination

**The Problem**
- Benchmark questions leak into model training data
- Models see the questions (or similar ones) during training
- They perform well because they've seen the answers, not because they're smarter

**Why It Happens**
- Hard to keep benchmarks completely secret
- Information spreads quickly in the AI community
- Training datasets are massive and may inadvertently include benchmark data

**Solutions**
- Keep benchmarks very secret
- Constantly rotate and update benchmarks
- Use fresh, never-before-seen questions

**Famous Example: Apple's Paper**
Apple researchers took a well-known benchmark and changed facts in the questions without changing the spirit of the question (e.g., changing names). Several models' performance dropped significantly, demonstrating contamination had occurred.

**Source**: https://arxiv.org/abs/2311.01964

### 2. Inconsistent Application

**Issues**
- Benchmarks aren't always run consistently
- Hardware differences can affect results
- Some benchmarks are self-reported by model providers
- Lack of standardization in testing conditions

**Result**: Hard to make direct comparisons

### 3. Narrow Scope

**The Problem**
- GPQA tests physics, chemistry, and biology specifically
- Good performance doesn't mean generally PhD-level intelligence
- Just means PhD-level in those specific areas

**Example**
Models can score 88% on GPQA but fail at seemingly simple tasks like:
- Drawing a chessboard one move from checkmate
- Combining visual and reasoning tasks

**Lesson**: Benchmarks test specific capabilities, not general intelligence

### 4. Limited Format

**Constraints**
- Most benchmarks use multiple choice
- Even creative benchmarks like MUSA have fixed structures
- Hard to test for nuance
- Need measurable right/wrong answers

**What's Missing**
- Open-ended creativity
- Subtle understanding
- Real-world messiness

### 5. Saturation

**The Cycle**
- New benchmark created
- Models quickly improve
- Within a year, models score 99%
- Benchmark becomes useless for differentiation

**Why It Happens**
- Models are improving rapidly
- Training techniques get better
- Inference-time techniques advance

**Solution**: Create harder benchmarks (like HLE)

### 6. Overfitting

**The Scenario**
Imagine Anthropic training the next Claude:
1. Create 4-5 candidate versions
2. Test each against GPQA
3. Pick the one with the best GPQA score
4. Repeat this process multiple times

**The Problem**
By repeatedly selecting the model that performs best on a specific metric, you're implicitly training for that metric. The selected model might just be the "lucky" one that randomly performs well on those specific questions, not actually the smartest model.

**The Test**
Change the benchmark questions slightly (like Apple did), and the "lucky" model's performance drops.

### 7. Evaluation Awareness (Emerging Concern)

**The Hypothesis**
Top frontier models might be able to "know" when they're being evaluated and adjust their behavior accordingly.

**Why It Matters**
Particularly concerning for testing "alignment" (following instructions):
- If a model knows it's being tested for alignment
- It might behave well during testing
- But behave differently when it thinks it's not being evaluated

**Current Status**
- Not yet proven or fully understood
- Security companies are researching this
- Anthropic has published interesting work on this topic
- One to watch for the future

**Source**: Anthropic's blog on AI safety and security

## Leaderboards: Your Essential Resources

Now let's explore the top leaderboards you should know about. These are invaluable resources for comparing models.

### 1. Artificial Analysis (⭐ Favorite)

**URL**: https://artificialanalysis.ai

**Why It's Great**
- Most comprehensive leaderboard available
- Incredibly clear presentation
- Multiple dimensions of comparison
- Independently analyzes LLMs

**Key Features**

**Intelligence Index**
- Combines 10 different evaluations
- Includes: MMLU Pro, GPQA, HLE, Live Code Bench, AIME, and others
- Weighted somewhat toward genetic coding
- Overall intelligence score

**Current Top Performers** (as of recording):
1. GPT-5 Codex (coding-optimized)
2. GPT-5 High (maximum reasoning mode)
3. Grok-4
4. Claude 4.5 Sonnet
5. Gemini 2.5 Pro
6. GPT-Opus-1 R20B (first open source model in top ranks!)

**Speed Comparisons**
- Tokens per second
- Latency (time to first token)
- Visual charts for easy comparison

**Price Analysis**
- Cost per million tokens
- Input vs. output pricing
- Cost-effectiveness ratios

**Historical Trends**
- Performance over time (back to November 2022)
- Shows rapid improvement trajectory
- Helps predict future trends

**Individual Benchmark Breakdowns**
- See performance on each specific benchmark
- Identify model strengths and weaknesses

**Why You Should Bookmark It**
This is the most useful single resource for model comparison. Check it regularly as it updates with new models.

### 2. Vellum

**URL**: https://www.vellum.ai/llm-leaderboard

**What Makes It Special**
- API costs and context windows side-by-side
- Easy comparison across providers
- Clean, organized presentation

**Key Features**
- Pricing comparison across all major providers
- Context window sizes
- Model capabilities overview

**Bonus**
Vellum also offers a full product for building AI applications into production. While exploring the leaderboard, check out their main platform for:
- Production AI workflows
- Prompt management
- Testing and evaluation tools

**Fun Fact**
The CEO of Vellum reached out after hearing about the leaderboard being mentioned in this course!

### 3. Scale AI (SEAL Leaderboards)

**URL**: https://scale.com/leaderboard

**About Scale**
- Well-known AI startup (now company)
- Partly owned by Meta
- Major player in AI infrastructure

**SEAL Leaderboards**
- SEAL = Scale Expert Assessment Leaderboards
- Very specialized, expert-focused benchmarks
- Multiple leaderboards for different capabilities
- High-quality, rigorous evaluation

**What Makes Them Different**
- Expert-level evaluation
- Domain-specific testing
- Professional assessment standards

### 4. Hugging Face Leaderboards

**URL**: https://huggingface.co/spaces (various leaderboard spaces)

**Background**
- Used to be THE go-to leaderboard
- The original "Open LLM Leaderboard" is no longer actively updated
- Possibly due to gaming/manipulation issues

**Current Status**
- Multiple specialized leaderboards still active
- Hosted as Hugging Face Spaces
- Focus on open source models
- Community-driven

**What to Explore**
- Various domain-specific leaderboards
- Open source model comparisons
- Community benchmarks

**Note**: While not as dominant as before, still valuable for open source model research.

### 5. Live Bench

**URL**: https://livebench.ai

**Special Focus**
- Addresses training data contamination
- Constantly rotating benchmarks
- Fresh questions that models haven't seen

**How It Works**
- Regularly updates with new questions
- Stays ahead of training data cycles
- Measures true raw performance

**Why It Matters**
- Most reliable for avoiding contamination
- True test of model capabilities
- Not influenced by training data leakage

**Use Case**
When you need to know a model's genuine capability, not just how well it memorized benchmark answers.

## Real-World Model Comparison Example

Let's look at a fun, practical example: Connect Four!

### The Connect Four Leaderboard

A custom-built leaderboard that tests models on playing Connect Four (the game where you drop coins to make four in a row).

**Why This Is Interesting**
- Easy task for humans
- Surprisingly hard for models
- Reveals limitations even in advanced models

**Why Models Struggle**
- Lots of visual information to process
- No loops (models just predict tokens)
- Diagonal pattern recognition is particularly hard

**How It Works**
1. Pick two models to play against each other
2. Models must explain their reasoning before each move
3. Forced reasoning mode improves performance
4. Even then, most models aren't at human level

**The Prompt Strategy**
Instead of just asking "which column?", the system asks models to:
1. Evaluate the board state
2. Identify threats
3. Identify opportunities
4. Explain strategy
5. Then choose a column

This forces reasoning and improves performance significantly.

**Key Insight**
Even with astronomical benchmark scores and "PhD-level" performance, most models can be beaten by humans at Connect Four. This illustrates the narrow scope of benchmarks and the difference between specialized intelligence and general capability.

**Try It Yourself**
The Connect Four leaderboard is available online and contributes to a community leaderboard. It's a great way to gain intuition about model reasoning.

## Introduction to RAG (Retrieval Augmented Generation)

Now we transition to one of the most important techniques in modern LLM applications: RAG.

### The Problem RAG Solves

**LLM Limitations**
1. **Knowledge Cutoff**: Models only know information up to their training date
2. **No Private Data**: Can't access your company's proprietary information
3. **Context Window Limits**: Can't fit all information in the prompt

**Examples of What LLMs Don't Know**
- Your company's internal documents
- Recent news (after training cutoff)
- Your personal data
- Proprietary databases

### What is RAG?

**RAG = Retrieval Augmented Generation**

**The Concept**
Instead of relying only on the model's training:
1. **Retrieve** relevant information from an external knowledge base
2. **Augment** the prompt with this information
3. **Generate** a response using both model knowledge and retrieved context

**Simple Analogy**
It's like an open-book exam instead of a closed-book exam. The model can "look up" information before answering.

### Simple RAG Example: Dictionary Lookup

Let's start with a basic (but limited) approach:

**The Knowledge Base**

```python
knowledge = {
    "lancaster": "Avery Lancaster, Co-founder and CEO of InsureElm",
    "thompson": "Maxine Thompson, Senior Data Engineer at InsureElm",
    "car": "CarElm - auto insurance product by InsureElm",
    "health": "HealthElm - health insurance product by InsureElm"
}
```

**The Lookup Function**

```python
def get_relevant_context(message):
    # Remove punctuation and convert to lowercase
    words = message.lower().split()
    
    # Look up each word
    relevant_context = []
    for word in words:
        if word in knowledge:
            relevant_context.append(knowledge[word])
    
    return relevant_context
```

**How It Works**
1. User asks: "Who is Lancaster?"
2. Function finds "lancaster" in the dictionary
3. Returns: "Avery Lancaster, Co-founder and CEO of InsureElm"
4. This context is added to the prompt
5. LLM generates response with this information

**The Pythonic Way**

```python
def get_relevant_context(message):
    words = message.lower().split()
    return [knowledge[word] for word in words if word in knowledge]
```

This list comprehension does the same thing more elegantly.

### Limitations of Simple Dictionary Lookup

**Problem 1: Exact Matching Only**
- "Who is Lancaster?" ✅ Works
- "Who is Avery?" ❌ Fails (only last names in dictionary)
- "Who is lancaster?" ✅ Works (we convert to lowercase)

**Problem 2: Typos Break Everything**
- "Who is Lancster?" ❌ Fails

**Problem 3: Synonyms Don't Work**
- "Who is the CEO?" ❌ Fails (no "CEO" key)

**Problem 4: Partial Matches Can Mislead**
- "Who is Alex Lancaster?" 
- Finds "Lancaster" key
- Returns info about Avery (wrong person!)

### The Conversation History Trap

**Interesting Behavior**
If you ask "Who is Lancaster?" first, then ask "Who is Avery?", the second question might work!

**Why?**
- The conversation history is included in the prompt
- Previous answer mentioned "Avery Lancaster"
- Model uses this context to answer the second question

**Testing Tip**
When testing RAG systems, always use fresh conversations to avoid this effect. Otherwise, you might think your retrieval is working when it's actually relying on conversation history.

### Why We Need Something Better

This simple dictionary approach is:
- **Brittle**: Breaks easily with small changes
- **Inflexible**: Can't handle variations
- **Limited**: Only exact string matching

**What We Need**
- Fuzzy matching
- Semantic understanding
- Ability to find relevant information even with different wording

**The Solution**: Vector embeddings and semantic search!

## Understanding Vectors and Embeddings

This is the big idea behind modern RAG systems.

### Two Types of LLMs

Up until now, we've only worked with one type of LLM. But there are actually two fundamentally different types:

### Type 1: Autoregressive LLMs (What We've Used)

**What They Do**
- Take an input sequence
- Predict the next token
- Repeat (add predicted token, predict next one)

**Examples**
- GPT (all versions)
- Claude
- Gemini
- DeepSeek
- Grok
- Basically every LLM you've heard of

**How They're Trained**
- Given input sequences and the correct next token
- Learn to predict what comes next
- Architecture supports sequential generation

**Why "Autoregressive"?**
- "Auto" = self
- "Regressive" = looking back
- Looks back at input to predict forward

### Type 2: Encoder LLMs (New to Us)

**Also Called**
- Encoders
- Embedding models
- Vector embedding models
- Autoencoders

**What They Do**
- Take a full input sequence
- Produce ONE output that represents the entire input
- Don't predict next tokens
- Output reflects the complete input meaning

**Use Cases**

**Classification**
- Sentiment analysis (positive/negative)
- Category detection
- Topic identification

**Vector Embeddings** (Our Focus)
- Convert text to a set of numbers
- Numbers represent the meaning
- Used for semantic search and similarity

**Examples**
- BERT (Google, 2018 - predates GPT!)
- OpenAI text-embedding-3-large
- OpenAI text-embedding-3-small
- all-MiniLM-L6-v2 (popular open source)

### Tokens vs. Vectors: Clearing Up Confusion

This is a common source of confusion, so let's clarify:

**Tokens = Inputs**
- Simple numeric representation of text
- Each word/fragment mapped to a number
- Just a way to get text into a model
- Models can't process words directly, need numbers
- Very simple mapping (like a lookup table)
- Examples: "the" = 1, "and" = 2, "cat" = 3

**Why Tokens?**
- Models do math (addition, multiplication)
- Can't do math with words
- Need numeric representation
- Token fragments (not full words or letters) work best

**Vectors = Outputs**
- Complex numeric representation of meaning
- Multiple numbers (100s or 1000s)
- Produced by a model after processing
- Capture semantic meaning
- Similar meanings → similar vectors

**The Flow**
1. Text → Tokens (simple conversion)
2. Tokens → Model (processing)
3. Model → Vectors (meaningful output)

**Pro Note**
Technically, vectors also exist inside models between layers. Tokens are converted to vectors internally. But for our purposes:
- Tokens = simple input representation
- Vectors = meaningful output representation

### What Are Vector Embeddings?

**The Concept**
A vector embedding is a set of numbers that represents the meaning of text.

**Example**
Text: "Ticket prices to London"
Vector: [0.2, 0.8, 0.5, -0.3, 0.7, 0.1, ...]

**Dimensions**
- Could be 3 numbers (like x, y, z coordinates)
- Usually 100s or 1000s of numbers
- More dimensions = more nuanced representation
- Think of it as a point in high-dimensional space

**Key Property**
Similar meanings → Similar vectors (close in vector space)

### How Vectors Represent Meaning

**Property 1: Proximity = Similarity**

Sentences with similar meanings have vectors close together in space:

```
"What are ticket prices to travel from New York to London?"
Vector: [0.2, 0.8, 0.5, ...]

"What's the price of a flight from JFK to Heathrow Airport?"
Vector: [0.21, 0.79, 0.51, ...]
```

Different words, same meaning, close vectors!

**How Models Learn This**
- Trained so outputs have this property
- Similar contexts → similar vectors
- Semantic meaning captured in numeric space

**Property 2: Vector Arithmetic (The Magic Part!)**

This was first discovered with Word2Vec (predecessor to modern embeddings).

**The Famous Example**

```
King - Man + Woman = Queen
```

**How This Works**
1. Each word has a vector (point in space)
2. You can add and subtract vectors (move in space)
3. Take "King", subtract "Man" concept, add "Woman" concept
4. The closest point to the result is "Queen"!

**More Examples**

```
Paris - France + England = London
```

**Why It's Amazing**
- You can manipulate meanings mathematically
- Remove one concept, add another
- Works with entire paragraphs (not just words) in modern encoders

**Modern Application**
With modern encoder models:
- Take a paragraph of text
- Remove some meaning (subtract vector)
- Add different meaning (add vector)
- Get a new paragraph with modified meaning

### Measuring Similarity

**Cosine Similarity**
The technical way to measure if vectors are "close":
- Not just distance between points
- Measures angle between vectors
- Values from -1 to 1
- 1 = identical meaning
- 0 = unrelated
- -1 = opposite meaning

**For Our Purposes**
Just remember: Similar vectors = similar meanings

## The Big Idea Behind RAG

Now we can put it all together!

### Traditional RAG Problem

**User asks**: "What are ticket prices to Heathrow?"

**Database has**: "Ticket prices to London: $500"

**Problem**: Simple string matching won't find "London" when searching for "Heathrow"

### Vector-Based RAG Solution

**Step 1: Encode the Question**
- User asks: "What are ticket prices to Heathrow?"
- Use encoder model to convert question → vector
- Vector represents the question's meaning

**Step 2: Prepare the Vector Database**
- Take all knowledge base documents
- Use same encoder to convert each → vector
- Store both the text AND its vector
- This is called a "vector data store" or "vector database"

**Step 3: Semantic Search**
- Compare question vector to all document vectors
- Find documents with closest vectors (similar meaning)
- Retrieve the actual text of those documents

**Step 4: Augment the Prompt**
- Take the user's original question (text)
- Add the retrieved relevant context (text)
- Send to the autoregressive LLM

**Step 5: Generate Response**
- LLM has both the question and relevant context
- Generates accurate answer using retrieved information

### The Key Insight

**Fuzzy Matching Through Meaning**
- Don't match strings
- Match meanings
- "Heathrow" and "London" are semantically similar
- Vectors capture this relationship
- Find relevant info even with different words

**Examples That Now Work**
- "Who is Avery?" → Finds "Avery Lancaster" info
- "Ticket prices to Heathrow?" → Finds "London" prices
- "Who is the CEO?" → Finds "Avery Lancaster, CEO" info
- "Lancster" (typo) → Still finds "Lancaster" (if vectors are close enough)

## Critical Clarification: Two Separate LLMs

This is crucial to understand and often confuses even experienced practitioners:

### The Encoder LLM (Left Side)

**Purpose**: Convert text to vectors for search
**Input**: Text (questions and documents)
**Output**: Vectors (numbers)
**Used For**: Semantic search only
**Example**: text-embedding-3-large

**What It Does NOT Do**
- Generate responses
- Chat with users
- Create new text

### The Autoregressive LLM (Right Side)

**Purpose**: Generate responses
**Input**: Natural language text (question + context)
**Output**: Natural language text (answer)
**Used For**: Answering questions
**Example**: GPT-4, Claude, Gemini

**What It Does NOT Do**
- Work with vectors directly
- Perform semantic search
- Encode text to vectors

### They Don't Talk to Each Other!

**The Flow**
1. Encoder converts question → vector
2. Vector search finds similar vectors in database
3. Retrieve TEXT associated with those vectors
4. Send TEXT (not vectors) to autoregressive LLM
5. Autoregressive LLM generates TEXT response

**Key Point**: The autoregressive LLM never sees vectors. It only receives and produces natural language text.

## RAG: A Practical Hack

Let's be honest about what RAG is:

### The Reality

**RAG is Empirical**
- Lots of trial and error
- Not a perfect science
- Constantly evolving

**RAG is a Hack**
- We're guessing what might be relevant
- Using vector similarity as a proxy for usefulness
- No guarantees it finds the best information

**RAG Has Many Hacks**
- Hack upon hack upon hack
- Techniques to improve retrieval
- Tricks to better select context
- Methods to handle edge cases

### Why It's OK

**It Works Well Enough**
- Better than no external knowledge
- Cheaper than massive context windows
- More flexible than fine-tuning

**It's Practical**
- Can update knowledge base easily
- No retraining required
- Scales to large knowledge bases

**It's Evolving**
- New techniques constantly emerging
- Community sharing improvements
- Research advancing rapidly

### What to Expect

**Coming Up**
- Many techniques to improve RAG
- Lots of experimentation
- Testing different approaches
- Measuring what works

**Mindset**
- Be empirical
- Test and measure
- Iterate and improve
- Accept imperfection

## Key Takeaways

### Model Selection

1. **No "Best" Model** - Only the right model for your specific task
2. **Consider Basics** - Parameters, cost, context window, license
3. **Check Benchmarks** - But understand their limitations
4. **Use Leaderboards** - Artificial Analysis, Vellum, Live Bench, etc.
5. **Test Your Use Case** - Benchmarks don't tell the whole story

### Benchmarks

1. **Useful Indicators** - Good starting point for comparison
2. **Hard Benchmarks** - GPQA, MMLU Pro, AIME, Live Code Bench, MUSA, HLE
3. **Serious Limitations** - Contamination, narrow scope, saturation, overfitting
4. **Take with Salt** - Use as guidance, not gospel

### RAG Fundamentals

1. **The Problem** - LLMs have knowledge cutoffs and can't access private data
2. **The Solution** - Retrieve relevant information and augment prompts
3. **Simple Approach** - Dictionary lookup (brittle, limited)
4. **Better Approach** - Vector embeddings and semantic search

### Vector Embeddings

1. **Two LLM Types** - Autoregressive (generate) and Encoder (embed)
2. **Vectors = Meaning** - Numbers that represent semantic content
3. **Similarity** - Close vectors = similar meanings
4. **Vector Math** - Can add/subtract concepts (King - Man + Woman = Queen)

### RAG Architecture

1. **Encoder LLM** - Converts text to vectors for search
2. **Vector Database** - Stores vectors + original text
3. **Semantic Search** - Finds similar vectors (not exact matches)
4. **Autoregressive LLM** - Generates responses using retrieved context
5. **Two Separate LLMs** - They don't communicate directly

### Practical Wisdom

1. **RAG is a Hack** - Empirical, imperfect, but practical
2. **Test and Iterate** - Lots of experimentation needed
3. **Conversation History** - Can affect testing, use fresh conversations
4. **Fuzzy Matching** - The key advantage over simple string matching

## Next Steps

**Coming Up**
- Hands-on RAG implementation
- LangChain framework
- Chroma vector database
- Building production RAG pipelines
- Advanced RAG techniques

**Practice**
- Explore the leaderboards mentioned
- Compare models for your specific use case
- Think about what knowledge bases you might want to build
- Consider how RAG could solve problems in your domain

**Resources to Explore**
- Artificial Analysis: https://artificialanalysis.ai
- Vellum: https://www.vellum.ai/llm-leaderboard
- Live Bench: https://livebench.ai
- OpenAI Embeddings Documentation: https://platform.openai.com/docs/guides/embeddings
- Research papers on benchmarks and scaling laws

## Conclusion

Today we covered a lot of ground! You now understand:
- How to evaluate and select models systematically
- The major benchmarks and their limitations
- Where to find reliable model comparisons
- The fundamental concept behind RAG
- How vector embeddings enable semantic search
- The architecture of a RAG system

This foundation prepares you for the hands-on RAG implementation coming next. Remember, RAG is one of the most important techniques in modern LLM applications, enabling models to access external knowledge and stay up-to-date without retraining.

The journey from simple dictionary lookups to sophisticated vector-based semantic search shows how we can augment LLM capabilities in practical, powerful ways. While RAG is admittedly a collection of hacks, it's a collection of hacks that works remarkably well in practice!

See you in the next session where we'll build our first real RAG system!

