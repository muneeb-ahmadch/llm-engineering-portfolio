# LLM Engineering - Day 12
## Comprehensive Lecture Notes (Continuation from Day 11)

---

# Introduction

Welcome back! In Day 11, we covered the fundamentals of Large Language Models - what they are, how to run them locally with Ollama, the difference between frontier and open source models, making basic API calls, and understanding system vs user prompts.

Today, we're going deeper. We'll understand what's really happening when we call an LLM, explore different types of models, learn about the transformer architecture, and build a real multi-step application.

---

# Part 1: Understanding the Chat Completions API

## What is the Chat Completions API?

The Chat Completions API is the standard way to interact with LLMs. It was invented by OpenAI and became so popular that every other provider adopted the same format.

The name tells you what it does:
- **Chat**: You provide a conversation (chat history)
- **Completions**: The model predicts what should come next (completes the chat)

Think of it like this: you give the model everything that's been said so far, and it predicts the most likely response. That's literally all it does - predict the most likely next words. The remarkable thing is that these predictions turn out to be genuinely useful answers!

**Source:** OpenAI API Documentation - https://platform.openai.com/docs/api-reference/chat

## What's Really Happening Behind the Scenes

When you write code like `openai.chat.completions.create()`, here's what actually happens:

1. Your code builds an HTTP request
2. That request goes to a URL (called an endpoint)
3. OpenAI's servers process it
4. A response comes back as JSON
5. The library converts that JSON to Python objects

The endpoint for OpenAI is:
```
POST https://api.openai.com/v1/chat/completions
```

Let me show you the raw version:

```python
import requests
import json

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

payload = {
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "user", "content": "Tell me a fun fact"}
    ]
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

This is exactly what the OpenAI library does for you - just packaged more nicely!

## The OpenAI Library: Just a Wrapper

Here's something important to understand: the OpenAI Python library is NOT running any AI. It's just a lightweight wrapper that:

1. Builds HTTP requests for you
2. Handles authentication
3. Converts JSON responses to Python objects
4. Provides nice error messages

When you write:
```python
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello"}]
)
```

It's the same as making that HTTP request manually - just cleaner code!

**Source:** OpenAI Python Library (open source) - https://github.com/openai/openai-python

## Using One Library for Multiple Providers

Here's something clever: because everyone adopted OpenAI's API format, you can use the same OpenAI library to call different providers!

**Calling Gemini with OpenAI's library:**
```python
from openai import OpenAI

# Create client pointing to Google's endpoint
gemini = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=google_api_key
)

# Same code structure!
response = gemini.chat.completions.create(
    model="gemini-2.5-pro",
    messages=[{"role": "user", "content": "Hello"}]
)
```

The only changes: different `base_url` and different `api_key`. The rest of your code stays exactly the same!

This works because Google, Anthropic, and others all created "OpenAI-compatible endpoints" - they accept requests in the same format.

---

# Part 2: Types of LLMs

## Base Models

A base model is the foundation - it's trained purely to predict what token comes next in a sequence.

**Characteristics:**
- No conversation training
- Just predicts the most likely continuation
- Like very advanced autocomplete

**Example behavior:**
```
Input: "The Eiffel Tower is located in"
Output: "Paris, France. It was constructed in 1889..."
```

You actually use a base model every day - the predictive text on your phone! When you type "Hello, how" and it suggests "are you?", that's a tiny base model predicting likely next words.

**When to use base models:**
- When you want to fine-tune a model for a specific task
- When you don't want chat-style behavior baked in
- For specialized applications

Before ChatGPT, people used base models by formatting prompts cleverly:
```
Q: What is the capital of France?
A: Paris

Q: What is 2+2?
A:
```

The model would predict "4" as the likely continuation!

## Chat Models (Instruct Models)

OpenAI had a breakthrough idea: what if we trained the model specifically for conversations?

**How they're made:**
1. Start with a base model
2. Train it on conversation data (user message → good response)
3. Use RLHF (Reinforcement Learning from Human Feedback)
4. Humans rate responses, model learns from ratings

**Result:** Models that understand:
- System prompts (overall instructions)
- User prompts (the question)
- How to respond helpfully

**Characteristics:**
- Fast responses
- Good for interactive use
- Better for creative content
- Sometimes considered "warmer" in tone

**Source:** "Training language models to follow instructions with human feedback" - OpenAI (2022)

## Reasoning Models (Thinking Models)

The next evolution: models that think before answering.

**How they work:**
1. Model generates a "thinking trace" first
2. Works through the problem step by step
3. Then provides the final answer

**Example:**
```
User: What is 17 x 23?

Model's thinking: "Let me break this down...
17 x 23 = 17 x 20 + 17 x 3
17 x 20 = 340
17 x 3 = 51
340 + 51 = 391"

Model's answer: "17 x 23 = 391"
```

**The "Wait" Trick (Budget Forcing)**

Researchers discovered something surprising: if you insert the word "Wait" into the model's thinking trace, it causes the model to reconsider and think deeper!

```
Model: "I should calculate 17 x 23..."
[Insert "Wait"]
Model: "Wait, let me verify this. Am I sure about my approach?"
```

This simple trick improves accuracy on complex problems. It's called "budget forcing" because you're forcing the model to spend more "thinking budget."

**Source:** "S1: Simple Test-Time Scaling" - Research Paper (2025)

## Hybrid Models

The latest models are "hybrid" - they can do both:
- Quick chat responses when appropriate
- Deep reasoning when needed

**Examples:**
- GPT-5
- Gemini 2.5 Pro
- Claude 4.5 Sonnet

These models automatically decide how much thinking to do based on the question. Ask "What's 2+2?" and it responds instantly. Ask a complex logic puzzle and it thinks carefully.

## Comparison Summary

| Aspect | Base | Chat | Reasoning |
|--------|------|------|-----------|
| Speed | Fast | Fast | Slower |
| Conversation | Poor | Great | Good |
| Problem Solving | Basic | Good | Excellent |
| Cost | Low | Low | Higher |
| Best For | Fine-tuning | Chat apps | Complex tasks |

---

# Part 3: The Transformer Architecture

## The Story of Transformers

In 2017, a team at Google published a paper titled "Attention Is All You Need." They probably didn't realize they were about to change the world.

**The Problem Before Transformers:**

Before transformers, we used architectures like LSTM (Long Short-Term Memory). LSTMs were powerful but had a critical flaw: they processed sequences step by step. You couldn't process word 5 until you'd processed words 1-4.

This made them:
- Slow to train
- Hard to parallelize
- Limited in how much data they could learn from

**The Transformer Solution:**

The transformer architecture introduced "attention" - a mechanism that lets the model look at ALL parts of the input simultaneously.

Instead of: "Process word 1, then word 2, then word 3..."
It does: "Look at all words at once, figure out which ones matter for each position"

This meant:
- Massively parallel processing
- Much faster training
- Could handle much more data
- Scaled beautifully with more compute

**Source:** "Attention Is All You Need" - Vaswani et al., Google (2017)

## The Rise of GPT

With transformers, OpenAI (a small company at the time) started building increasingly large models:

| Year | Model | Parameters | Notes |
|------|-------|------------|-------|
| 2018 | GPT-1 | 117 million | First GPT |
| 2019 | GPT-2 | 1.5 billion | "Too dangerous to release" |
| 2020 | GPT-3 | 175 billion | Game changer |
| 2022 | ChatGPT | GPT-3.5 | 100M users in 2 months |
| 2023 | GPT-4 | 1.76 trillion | Multimodal |
| 2024+ | GPT-5 | Unknown | Current state of art |

## Emergent Intelligence

Here's what surprised everyone: at some point, these models started showing behaviors no one explicitly programmed.

The models are trained to predict likely next tokens. That's it. But somehow, predicting tokens well enough leads to:
- Answering questions correctly
- Solving math problems
- Writing code
- Reasoning about logic

We call this "emergent intelligence" - intelligence that emerges from scale. We understand HOW it works (the math, the architecture), but we're still puzzled about WHY it works so well.

The models aren't trying to be truthful - they're trying to predict likely text. Yet somehow, likely text often happens to be true text!

---

# Part 4: Parameters Explained

## What Are Parameters?

Parameters are the numerical values inside the neural network that got adjusted during training. Think of them as the model's "knowledge" encoded in numbers.

**Analogy:**
- More parameters = More "brain cells"
- More capacity to store patterns
- More ability to learn complex relationships

## The Scale of Parameters

| Model | Parameters |
|-------|------------|
| GPT-1 | 117 million |
| GPT-2 | 1.5 billion |
| GPT-3 | 175 billion |
| GPT-4 | 1.76 trillion |
| Llama 3.2 | 3 billion |
| DeepSeek | 671 billion |

Remember: GPT-4 has 1,760,000,000,000 parameters. Each one is a number that influences predictions!

## Getting Better with Less

Interestingly, we've gotten much better at efficiency. Gemma with 270 million parameters is more capable than GPT-2 with 1.5 billion, because we've learned better training techniques.

This is why DeepSeek made headlines - they achieved near-GPT-4 performance with much less training cost by being more efficient.

## Two Ways to Improve: Scaling

**Training Time Scaling:**
- Make the model bigger
- More parameters
- More training data
- Expensive upfront, benefits everyone

**Inference Time Scaling:**
- Same model
- Better prompts
- More reasoning (thinking tokens)
- RAG (adding context)
- Expensive per-use

Both approaches work! Modern LLM engineering uses both.

---

# Part 5: Tokens and Tokenization

## What Are Tokens?

Tokens are the chunks of text that models actually process. They're not characters and they're not exactly words - they're something in between.

**Why tokens?**

Early attempts used characters:
- Simple (only ~100 possible characters)
- But model had to learn spelling AND meaning - too hard

Then we tried words:
- Made sense conceptually
- But vocabulary exploded (names, technical terms, etc.)

Tokens are the middle ground:
- Common words = 1 token
- Rare words = multiple tokens
- Can represent anything by combining tokens

## Examples of Tokenization

Using GPT's tokenizer:

| Text | Tokens |
|------|--------|
| "Hello" | 1 token |
| "AI" | 1 token |
| "Banana" | 1 token |
| "Unbelievable" | 3 tokens (Un-believ-able) |
| "3.14159" | 3 tokens (3, .141, 59) |

Numbers are particularly interesting - they get split into 3-digit chunks:
```
"123456789" → ["123", "456", "789"]
```

This is why early GPT models struggled with 4+ digit arithmetic - the numbers were split across tokens!

## Token Counts

**Rule of thumb:**
- 1 token ≈ 4 characters
- 1 token ≈ 0.75 words
- 1000 tokens ≈ 750 words

**Reference point:**
The Complete Works of Shakespeare ≈ 900,000 words ≈ 1.2 million tokens

So when you see pricing "per million tokens," think "roughly the complete works of Shakespeare."

**Source:** OpenAI Tokenizer - https://platform.openai.com/tokenizer

---

# Part 6: The Illusion of Memory

## A Critical Understanding

Here's something that trips up many people: **Every API call is completely stateless.**

When you call the API, the model has NO memory of previous calls. It doesn't know you called it 30 seconds ago. It doesn't remember your name from the last message.

Let me demonstrate:

**First call:**
```python
messages = [{"role": "user", "content": "My name is John"}]
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
# Response: "Nice to meet you, John!"
```

**Second call:**
```python
messages = [{"role": "user", "content": "What's my name?"}]
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
# Response: "I don't have access to your name."
```

Wait, what? It just said "Nice to meet you, John!" Why doesn't it remember?

## How We Create "Memory"

The secret: **we pass the entire conversation history with every call!**

```python
messages = [
    {"role": "user", "content": "My name is John"},
    {"role": "assistant", "content": "Nice to meet you, John!"},
    {"role": "user", "content": "What's my name?"}
]
response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
# Response: "Your name is John!"
```

Now it "remembers" - but only because we told it what was said before!

## The Five Key Points

1. **Every call is stateless** - No persistent memory
2. **We pass full history** - Every message, every time
3. **It creates an illusion** - Feels like memory, but it's not
4. **Costs accumulate** - You pay for all those input tokens
5. **Context limits apply** - History can't exceed context window

This is exactly how ChatGPT works. The engineers at OpenAI aren't using magic - they're storing your messages and sending them all back with each request.

---

# Part 7: Context Windows

## What Is the Context Window?

The context window is the maximum number of tokens a model can process in one call. This includes:
- System prompt
- All conversation history
- Your new message
- The generated response

## Context Window Sizes

| Model | Context Window |
|-------|---------------|
| GPT-5 | 400,000 tokens |
| GPT-5 Nano | 200,000 tokens |
| Claude 3 | 200,000 tokens |
| Gemini 2.5 | 1,000,000 tokens |
| Llama 3.2 | 128,000 tokens |

Gemini with 1 million tokens can process almost the entire Complete Works of Shakespeare in one prompt!

## Why Context Windows Matter

1. **Conversation length**: Long chats eventually hit the limit
2. **RAG applications**: More context = more information to reference
3. **Document analysis**: Can you fit the whole document?
4. **Code assistance**: How much code can the model see?

When you exceed the context window, the API returns an error. You need to truncate or summarize history.

---

# Part 8: API Costs

## How Pricing Works

Most APIs charge per token:
- **Input tokens**: What you send (cheaper)
- **Output tokens**: What model generates (more expensive)

**GPT-5 Pricing:**
- Input: $1.25 per million tokens
- Output: $10 per million tokens

**GPT-5 Nano Pricing:**
- Input: $0.05 per million tokens
- Output: $0.40 per million tokens

## Understanding the Scale

Let's put this in perspective:
- Simple chat message: ~50 tokens
- That's 0.00005 million tokens
- Cost: Tiny fractions of a cent

For individual use, costs are negligible. For applications serving millions of users, they add up quickly.

## Hidden Costs to Watch

1. **Conversation history**: You pay for all previous messages every time
2. **Reasoning tokens**: Thinking costs money (even if hidden)
3. **Agent loops**: Multiple calls can multiply costs

**Source:** Model pricing at https://openai.com/pricing

---

# Part 9: Building Multi-Step Applications

## The Pattern

Real applications often need multiple LLM calls:
1. Call #1: Analyze or extract information
2. Process results in code
3. Call #2: Generate or transform based on results

## Example: Company Brochure Generator

Let's build something that:
1. Takes a company's website URL
2. Finds relevant pages (about, careers, etc.)
3. Generates a marketing brochure

### Step 1: Get Links and Select Relevant Ones

First, we scrape all links from the website. Then we ask GPT to identify which are relevant:

```python
link_system_prompt = """You are provided with a list of links 
from a website. Decide which are relevant for a company brochure.
Respond in JSON format like:
{
  "links": [
    {"type": "about page", "url": "https://..."},
    {"type": "careers", "url": "https://..."}
  ]
}
"""
```

This is called **one-shot prompting** - we give one example of the output format we want.

### Step 2: Force JSON Output

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_format={"type": "json_object"}
)
```

The `response_format` parameter constrains the model to only output valid JSON. This works at the token-generation level - invalid JSON tokens get blocked!

### Step 3: Fetch Relevant Pages and Generate Brochure

```python
brochure_system_prompt = """You are an assistant that creates 
marketing brochures. Based on the company website content provided,
create a compelling brochure for customers, investors, and recruits.
Respond in markdown."""
```

### Step 4: Stream the Results

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    stream=True  # Enable streaming!
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="")
```

Streaming gives that typewriter effect - results appear as they're generated!

---

# Part 10: Practical Tips

## Prompt Engineering Best Practices

1. **Be specific**: Tell the model exactly what you want
2. **Give examples**: One-shot or multi-shot prompting
3. **Specify format**: "Respond in JSON" or "Use markdown"
4. **Iterate constantly**: Refine prompts based on results

## When to Use What

| Task | Model Type | Streaming |
|------|-----------|-----------|
| Quick chat | Chat model | Yes |
| Complex reasoning | Reasoning model | Optional |
| Data extraction | Chat + JSON format | No |
| Content generation | Chat model | Yes |

## Cost Optimization

1. Use smaller models when possible (GPT-4o-mini vs GPT-5)
2. Truncate conversation history appropriately
3. Use caching when sending similar prompts
4. Consider open source for high-volume applications

---

# Summary

## What We Covered Today

1. **Chat Completions API**: Just HTTP calls wrapped nicely
2. **Model Types**: Base, Chat, Reasoning, Hybrid
3. **Transformers**: The architecture that changed everything
4. **Parameters**: Model "knowledge" in numbers
5. **Tokens**: How text gets processed
6. **Memory Illusion**: Stateless calls, we provide history
7. **Context Windows**: Limits on how much we can send
8. **API Costs**: Per-token pricing
9. **Multi-Step Apps**: Chaining LLM calls for complex tasks

## Key Takeaways

1. There's no magic - it's HTTP requests and token predictions
2. The model doesn't remember anything - we manage that
3. Costs scale with token usage - design accordingly
4. Multiple LLM calls can solve complex problems
5. Experimentation is key - iterate on your prompts

---

# Resources

- OpenAI Documentation: https://platform.openai.com/docs
- OpenAI Tokenizer: https://platform.openai.com/tokenizer
- "Attention Is All You Need" paper: https://arxiv.org/abs/1706.03762
- Vellum Leaderboard (costs/context windows): https://vellum.ai/llm-leaderboard

---

*Next session: We'll dive into tool calling, agents, and building interactive chat interfaces!*

