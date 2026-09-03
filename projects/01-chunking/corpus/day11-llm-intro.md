# Introduction to LLM Engineering
## Comprehensive Lecture Notes

---

# Part 1: Introduction to Large Language Models

## What is a Large Language Model?

Welcome to your journey into LLM Engineering! Today we will learn how to work with Large Language Models - both running them locally on your computer and calling them through cloud APIs.

A Large Language Model (LLM) is an artificial intelligence system that has been trained on enormous amounts of text data from the internet, books, articles, and other written sources. The word "large" refers to the number of parameters (think of these as adjustable settings) in the model - typically ranging from hundreds of millions to hundreds of billions.

**How do LLMs work?**

At their core, LLMs are prediction machines. They predict what word (or more precisely, what "token") should come next given all the previous words. When you type "The capital of France is", the model calculates probabilities for what word should follow and picks "Paris" because it's the most likely continuation based on its training.

This simple concept of "next word prediction" leads to surprisingly intelligent behavior when scaled up with:
- **Massive training data** - Trillions of words from the internet
- **Enormous model size** - Billions of parameters to learn patterns
- **Powerful computing** - Thousands of GPUs training for weeks

**Think of it like this:** If you've read millions of books and someone asks you "The capital of France is...", you'd naturally complete it with "Paris." LLMs do the same thing, but at an unprecedented scale.

**Historical Context:**
- 2017: Google introduces the Transformer architecture (the "T" in GPT)
- 2018: OpenAI releases GPT-1 with 117 million parameters
- 2020: GPT-3 launches with 175 billion parameters - a breakthrough moment
- 2022: ChatGPT released and gains 100 million users in 2 months
- 2023-2024: Explosion of models from multiple companies

**Source:** "Attention Is All You Need" - Vaswani et al., Google (2017)
**Source:** "Language Models are Few-Shot Learners" - OpenAI (2020)

---

# Part 2: Running LLMs Locally with Ollama

## What is Ollama?

Ollama is a free, open-source tool that makes it incredibly easy to download and run large language models directly on your personal computer. The name is a playful reference to Meta's "Llama" model series.

**Why is local execution important?**

When you use ChatGPT or Claude, your prompts travel to servers owned by OpenAI or Anthropic. With Ollama, everything happens on your machine:

| Aspect | Cloud (ChatGPT) | Local (Ollama) |
|--------|-----------------|----------------|
| Privacy | Data sent to company servers | Data stays on your computer |
| Cost | Pay per use (tokens) | Completely free |
| Internet | Required always | Only for download |
| Speed | Depends on server load | Depends on your hardware |
| Model Access | Limited to what they offer | Any open source model |

**Source:** https://ollama.com

## Installation Process

### Step 1: Download Ollama

1. Open your web browser and go to **ollama.com**
2. Click the prominent **Download** button
3. Select your operating system:
   - **Windows**: Download the .exe installer
   - **Mac**: Download the .dmg file
   - **Linux**: Use the curl command provided

### Step 2: Install the Application

**On Windows:**
- Double-click the downloaded .exe file
- Follow the installation wizard
- Ollama will install and start automatically

**On Mac:**
- Open the downloaded .dmg file
- Drag Ollama to your Applications folder
- Open Ollama from Applications (you may need to allow it in Security settings)

**On Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Step 3: Verify Installation

Open your terminal:
- **Windows**: Press `Win + X`, select "Windows PowerShell"
- **Mac**: Press `Cmd + Space`, type "Terminal", press Enter
- **Linux**: Press `Ctrl + Alt + T`

Type the following command:
```bash
ollama
```

You should see a help menu showing available commands like `run`, `pull`, `list`, etc.

## Understanding Model Parameters

When you see a model like `gemma3:270m` or `llama3:8b`, the number refers to **parameters**.

**What are parameters?**

Parameters are the numerical values (weights) that the model learned during training. They represent the model's "knowledge." More parameters generally means:
- More knowledge capacity
- Better understanding of nuance
- Higher quality responses
- BUT: More memory required and slower responses

**Size Comparison:**

| Size Category | Parameter Count | RAM Required | Example Models |
|---------------|-----------------|--------------|----------------|
| Tiny | 270M - 1B | 1-2 GB | gemma3:270m |
| Small | 1B - 3B | 2-4 GB | llama3.2:1b, phi3:mini |
| Medium | 7B - 8B | 8-12 GB | llama3:8b, mistral:7b |
| Large | 13B - 30B | 16-32 GB | llama3:13b, mixtral |
| Very Large | 70B+ | 40GB+ | llama3:70b (needs powerful GPU) |

**Rule of thumb:** You need approximately 1GB of RAM for every 1 billion parameters, plus some overhead.

## Running Your First Model

Let's start with a small model that will work on most computers:

```bash
ollama run gemma3:270m
```

**What happens when you run this command?**

1. **Check**: Ollama checks if the model is already downloaded
2. **Download**: If not, it downloads the model file (called a GGUF file)
3. **Load**: The model is loaded into your computer's memory
4. **Ready**: You see a `>>>` prompt indicating you can start chatting

**Example conversation:**
```
>>> Hello! What can you help me with?
Hello! I'm a helpful AI assistant. I can help you with:
- Answering questions
- Writing and editing text
- Explaining concepts
- Problem-solving
- And much more! What would you like to know?

>>> What's the weather like?
I don't have access to real-time information like current weather. 
I'm a language model that works with text. For weather, 
I'd recommend checking a weather app or website.

>>> Goodbye!
Goodbye! Feel free to come back anytime you have questions.
```

**To exit:** Press `Ctrl + D` (or `Ctrl + C` on Windows)

## Trying Different Models

Ollama supports dozens of models. Here are some worth trying:

**For general conversation:**
```bash
ollama run phi3           # Microsoft's efficient model (3.8B)
ollama run llama3:8b      # Meta's popular model (8B)
ollama run mistral        # Mistral AI's model (7B)
```

**For coding help:**
```bash
ollama run codellama      # Specialized for code
ollama run deepseek-coder # Another coding specialist
```

**For reasoning tasks:**
```bash
ollama run qwen2.5        # Alibaba's capable model
```

**Source:** https://ollama.com/library (full list of available models)

---

# Part 3: Understanding the AI Model Landscape

## Frontier Models (Closed Source)

The term "frontier models" refers to the most capable AI models available - typically created by well-funded AI labs and accessed through paid APIs or subscriptions.

### The Big Four Frontier Labs

**1. OpenAI - GPT Series**

OpenAI is the company that brought AI into mainstream awareness with ChatGPT. Their GPT (Generative Pre-trained Transformer) models are considered the gold standard.

- **ChatGPT**: The consumer product (website/app)
- **GPT-4, GPT-4o, GPT-5**: The underlying models
- **Strengths**: Excellent at following instructions, great for coding, strong reasoning
- **How to access**: ChatGPT Plus subscription ($20/month) or API
- **Source**: https://openai.com

**2. Anthropic - Claude Series**

Founded by former OpenAI researchers, Anthropic created Claude with a focus on safety and helpfulness.

- **Claude Haiku**: Fastest, cheapest (good for simple tasks)
- **Claude Sonnet**: Balanced performance and cost
- **Claude Opus**: Most capable (complex reasoning)
- **Strengths**: Thoughtful responses, good at nuanced tasks, strong at analysis
- **How to access**: Claude.ai free tier or API
- **Source**: https://anthropic.com

**3. Google - Gemini Series**

Google's entry into the frontier model space, integrating deeply with Google's ecosystem.

- **Gemini Nano**: On-device (phones)
- **Gemini Pro**: General use
- **Gemini Ultra**: Most powerful
- **Strengths**: Multimodal (text + images), integration with Google services
- **How to access**: Gemini app, Google AI Studio, Vertex AI
- **Source**: https://deepmind.google/technologies/gemini

**4. xAI - Grok**

Elon Musk's AI company, with Grok available through X (Twitter).

- **Grok-1, Grok-2**: Main model versions
- **Strengths**: Real-time information from X, less restrictive responses
- **How to access**: X Premium subscription
- **Source**: https://x.ai

### Comparison Table

| Model | Best For | Cost | Speed |
|-------|----------|------|-------|
| GPT-4o | General excellence | $$$ | Fast |
| Claude Sonnet | Writing, analysis | $$ | Medium |
| Gemini Pro | Google integration | $$ | Fast |
| Grok | Current events | $$ | Fast |

## Open Source Models

Open source models can be downloaded and run freely. The code and weights are publicly available.

### Why Open Source Matters

1. **Transparency**: You can see how the model works
2. **Privacy**: Your data never leaves your computer
3. **Cost**: No per-token charges
4. **Customization**: You can fine-tune on your own data
5. **No censorship**: Full control over model behavior

### Major Open Source Models

**1. Llama (Meta)**

Meta (Facebook) released Llama to democratize AI access. It's become the most popular open source LLM.

- **Llama 3.2**: Smaller versions (1B, 3B) - great for local use
- **Llama 4**: Latest, most capable
- **Why it matters**: Proved that open source can match closed source quality
- **Source**: https://llama.meta.com

**2. Mistral / Mixtral (Mistral AI)**

French startup that punches above its weight with efficient models.

- **Mistral 7B**: Compact but powerful
- **Mixtral 8x7B**: Uses "Mixture of Experts" - only activates relevant parts
- **Why it matters**: Excellent performance-to-size ratio
- **Source**: https://mistral.ai

**3. Qwen (Alibaba Cloud)**

Chinese tech giant Alibaba's contribution to open source AI.

- **Qwen 2.5**: Latest version, multiple sizes
- **Why it matters**: Often overlooked but extremely capable
- **Source**: https://github.com/QwenLM/Qwen

**4. DeepSeek (DeepSeek AI)**

Made headlines by achieving near-GPT-4 performance at a fraction of the training cost.

- **DeepSeek V2/V3**: Main versions
- **Why it matters**: Showed efficiency is possible (trained for ~$4M vs OpenAI's $100M+)
- **Training innovation**: Used more efficient architectures
- **Source**: https://www.deepseek.com

**5. Gemma (Google)**

Google's open source contribution, the smaller sibling of Gemini.

- **Gemma 2**: Current version
- **Sizes**: From 270M to 27B parameters
- **Why it matters**: Google's validation of open source approach
- **Source**: https://ai.google.dev/gemma

---

# Part 4: Three Ways to Use LLMs

Understanding the different ways to interact with LLMs helps you choose the right approach for your needs.

## Method 1: Consumer Products

**What are they?**
Complete applications with user interfaces built around LLMs.

**Examples:**
- ChatGPT (chat.openai.com)
- Claude (claude.ai)
- Gemini (gemini.google.com)
- Perplexity (perplexity.ai)

**Characteristics:**
- Easy to use - just type and get responses
- Additional features: memory, web search, file upload
- Monthly subscription pricing
- Limited customization

**Best for:** Everyday users, quick tasks, exploration

## Method 2: Cloud APIs

**What are they?**
Programming interfaces that let you send requests to models running on company servers.

**Examples:**
- OpenAI API
- Anthropic API
- Google Vertex AI
- Azure OpenAI
- AWS Bedrock

**Characteristics:**
- Pay per use (typically per 1000 tokens)
- Full programmatic control
- Can integrate into your applications
- Requires coding knowledge

**Pricing Example (OpenAI, approximate):**

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| GPT-4o | $5.00 | $15.00 |
| GPT-4o-mini | $0.15 | $0.60 |
| GPT-3.5-turbo | $0.50 | $1.50 |

**Best for:** Developers, applications, automation

## Method 3: Local Inference

**What is it?**
Running models directly on your own computer.

**Tools:**
- **Ollama**: Easy to use, optimized for consumer hardware
- **Hugging Face Transformers**: More control, Python library
- **llama.cpp**: Maximum performance, C++ based
- **LM Studio**: GUI application for local models

**Characteristics:**
- One-time download, unlimited free use
- Complete privacy
- Requires decent hardware
- Smaller model selection (limited by your hardware)

**Best for:** Privacy-conscious users, experimentation, offline use

---

# Part 5: Making Your First API Call to OpenAI

## Setting Up Your Environment

### Step 1: Get an OpenAI API Key

1. Go to **https://platform.openai.com**
2. Click "Sign Up" or "Log In"
3. Navigate to **API Keys** section (in the left sidebar)
4. Click **"Create new secret key"**
5. Give it a name (e.g., "LLM Learning")
6. **IMPORTANT**: Copy the key immediately! You won't see it again.

Your key will look like: `sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx`

### Step 2: Secure Your API Key

**Never put API keys directly in your code!** If you share your code or upload it to GitHub, others could steal your key and charge costs to your account.

**The solution: Environment variables**

Create a file called `.env` (note the dot at the beginning) in your project folder:

```
OPENAI_API_KEY=sk-proj-your-actual-key-here
```

**Important notes:**
- No spaces around the `=` sign
- No quotes around the key
- Save the file in the same folder as your code
- Add `.env` to your `.gitignore` file

### Step 3: Load the Key in Python

```python
# Import necessary libraries
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

# Get the API key
api_key = os.getenv("OPENAI_API_KEY")

# Verify it loaded correctly
if api_key and api_key.startswith("sk-"):
    print("API key loaded successfully!")
else:
    print("Error: Check your .env file")
```

## The Chat Completions API

This is the primary way to interact with GPT models programmatically.

### Basic Structure

```python
from openai import OpenAI

# Create a client (automatically uses OPENAI_API_KEY from environment)
client = OpenAI()

# Make an API call
response = client.chat.completions.create(
    model="gpt-4o-mini",  # Which model to use
    messages=[            # The conversation
        {"role": "user", "content": "Hello!"}
    ]
)

# Extract the response text
answer = response.choices[0].message.content
print(answer)
```

### Understanding the Response

When you make an API call, you get back a response object with this structure:

```python
{
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1699000000,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 15,
        "total_tokens": 25
    }
}
```

**Key parts:**
- `choices[0].message.content` - The actual response text
- `usage` - How many tokens were used (affects billing)
- `finish_reason` - Why the model stopped (usually "stop")

**Source:** OpenAI API Documentation - https://platform.openai.com/docs

---

# Part 6: System Prompts vs User Prompts

## The Messages Format

OpenAI's API uses a specific format for conversations: a list of message dictionaries, each with a `role` and `content`.

### Three Types of Roles

**1. System Role**
Sets up the AI's behavior, personality, and constraints.
```python
{"role": "system", "content": "You are a helpful assistant."}
```

**2. User Role**
Represents messages from the human user.
```python
{"role": "user", "content": "What is the capital of France?"}
```

**3. Assistant Role**
Represents previous AI responses (for multi-turn conversations).
```python
{"role": "assistant", "content": "The capital of France is Paris."}
```

## System Prompts in Depth

The system prompt is your primary tool for controlling AI behavior. It's processed once at the beginning and shapes all subsequent responses.

### What to Include in a System Prompt

1. **Role/Identity**: Who the AI should pretend to be
2. **Task**: What it should do
3. **Tone**: How it should communicate
4. **Constraints**: What it should avoid
5. **Format**: How to structure responses

### Examples of Effective System Prompts

**Example 1: Customer Service Bot**
```python
system_prompt = """You are a friendly customer service representative 
for TechGadgets Inc. Your role is to:
- Answer questions about our products
- Help with order issues
- Be patient and understanding
- Never discuss competitor products
- Always end by asking if there's anything else you can help with

Keep responses concise but helpful."""
```

**Example 2: Code Reviewer**
```python
system_prompt = """You are an experienced software engineer reviewing code.
For each piece of code shown:
1. Identify any bugs or issues
2. Suggest improvements
3. Rate the code quality (1-10)
4. Provide a brief explanation

Be constructive, not harsh. Focus on teaching."""
```

**Example 3: Language Tutor**
```python
system_prompt = """You are a Spanish language tutor for beginners.
- Use simple vocabulary
- Provide translations in parentheses
- Correct mistakes gently
- Encourage the student
- Gradually introduce new words
- Keep responses short and conversational"""
```

## The Power of Prompt Engineering

Small changes to your system prompt can dramatically change outputs:

**Experiment: Same question, different prompts**

Question: "What is 2 + 2?"

| System Prompt | Response |
|--------------|----------|
| "You are a helpful assistant" | "2 + 2 equals 4." |
| "You are a snarky assistant" | "Oh wow, a real brain-teaser. It's 4." |
| "You are a kindergarten teacher" | "Great question! Let's count: 1, 2... and 1, 2 more... that's 4!" |
| "Respond only in haiku format" | "Two plus two makes four / Mathematics so simple / Numbers never lie" |

---

# Part 7: Building a Website Summarizer

Let's combine everything we've learned to build something practical.

## The Concept

We'll create a tool that:
1. Takes any website URL
2. Fetches the webpage content
3. Sends it to GPT with instructions to summarize
4. Displays a nicely formatted summary

## The Components

### 1. Web Scraping

We use Python to fetch webpage content:

```python
import requests
from bs4 import BeautifulSoup

def fetch_website(url):
    """Fetch and clean website content."""
    # Get the webpage
    response = requests.get(url)
    
    # Parse HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove scripts and styles
    for element in soup(['script', 'style', 'nav', 'footer']):
        element.decompose()
    
    # Get text content
    text = soup.get_text(separator='\n')
    
    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return '\n'.join(lines)
```

### 2. System Prompt for Summarization

```python
system_prompt = """You are an assistant that analyzes website content 
and provides clear, concise summaries.

Guidelines:
- Focus on the main content, ignore navigation/ads
- Highlight key points and takeaways
- Use bullet points for clarity
- If there's news or announcements, include them
- Keep the summary under 200 words
- Respond in markdown format"""
```

### 3. The API Call

```python
def summarize_website(url):
    """Summarize a website's content."""
    # Fetch content
    content = fetch_website(url)
    
    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize this website:\n\n{content}"}
    ]
    
    # Call API
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    return response.choices[0].message.content
```

### 4. Display the Result

```python
from IPython.display import Markdown, display

# Use it!
url = "https://example.com"
summary = summarize_website(url)
display(Markdown(summary))
```

## Customizing the Output

By modifying the system prompt, you can change what kind of summary you get:

**Snarky Summary:**
```python
system_prompt = """You are a snarky critic reviewing websites. 
Provide a humorous summary that pokes gentle fun at the content 
while still conveying the main points."""
```

**Executive Summary:**
```python
system_prompt = """You are a business analyst. Provide an executive 
summary suitable for C-level executives. Focus on business value, 
key metrics, and actionable insights. Be professional and concise."""
```

**Translation:**
```python
system_prompt = """You are a translator. Summarize the website content 
in Spanish. Maintain a professional tone suitable for business communication."""
```

---

# Part 8: Practical Business Applications

The patterns we've learned apply to countless real-world scenarios.

## Application 1: Email Assistant

**Use Case:** Automatically generate email subject lines and summaries

```python
system_prompt = """You are an email assistant. For each email provided:
1. Suggest 3 subject line options
2. Provide a one-sentence summary
3. Identify the action required (if any)
4. Rate urgency (Low/Medium/High)"""
```

## Application 2: Content Repurposing

**Use Case:** Transform long content into social media posts

```python
system_prompt = """You are a social media manager. Transform the 
given content into:
1. A Twitter/X post (under 280 characters)
2. A LinkedIn post (professional tone, 2-3 paragraphs)
3. An Instagram caption (engaging, with emoji suggestions)"""
```

## Application 3: Meeting Notes Processor

**Use Case:** Extract action items from meeting transcripts

```python
system_prompt = """You are a meeting assistant. From the transcript:
1. List all action items with owners and deadlines
2. Summarize key decisions made
3. Note any open questions
4. Identify follow-up meetings needed"""
```

## Application 4: Customer Feedback Analyzer

**Use Case:** Categorize and summarize customer reviews

```python
system_prompt = """You are a customer insights analyst. For each review:
1. Sentiment: Positive/Neutral/Negative
2. Main topics mentioned
3. Specific praise or complaints
4. Suggested response approach"""
```

---

# Part 9: Key Terminology Glossary

| Term | Definition | Example |
|------|------------|---------|
| **LLM** | Large Language Model - AI trained to generate text | GPT-4, Claude, Llama |
| **Token** | A chunk of text (roughly 4 characters) | "Hello" = 1 token |
| **Prompt** | Input text given to the model | "What is AI?" |
| **Completion** | The model's generated response | "AI is..." |
| **Parameters** | Model's learned weights | GPT-4 has ~1.7T |
| **Inference** | Running a model to get output | Getting a response |
| **Fine-tuning** | Training further on specific data | Teaching medical terms |
| **RAG** | Retrieval Augmented Generation | Search + Generate |
| **Embedding** | Converting text to numbers | For similarity search |
| **Context Window** | How much text model can see | 8K - 128K tokens |
| **Temperature** | Randomness in output | 0=deterministic, 1=creative |
| **API** | Application Programming Interface | OpenAI's endpoint |
| **Hallucination** | Model making up false info | Citing fake sources |

---

# Part 10: Summary and Next Steps

## What We Covered Today

1. **LLM Fundamentals**: What they are and how they work
2. **Ollama**: Running models locally on your computer
3. **Model Landscape**: Frontier vs open source models
4. **API Usage**: Making calls to OpenAI
5. **Prompt Engineering**: System vs user prompts
6. **Practical Application**: Building a website summarizer

## Key Takeaways

1. **LLMs are prediction machines** - They predict the next token based on patterns learned from training data

2. **You have options** - Cloud APIs for power, local models for privacy, products for ease of use

3. **System prompts are powerful** - Small changes dramatically affect output

4. **Start simple, iterate** - Begin with basic prompts and refine based on results

## Practice Exercises

1. **Ollama Exploration**: Try 3 different models and compare responses to the same question

2. **Prompt Engineering**: Create 5 different system prompts and observe how they change outputs

3. **Build Something**: Create your own tool (email summarizer, code explainer, etc.)

4. **Cost Optimization**: Compare GPT-4o vs GPT-4o-mini for your use case

## Resources for Further Learning

**Official Documentation:**
- OpenAI: https://platform.openai.com/docs
- Anthropic: https://docs.anthropic.com
- Ollama: https://ollama.com/library

**Learning Resources:**
- OpenAI Cookbook: https://cookbook.openai.com
- Hugging Face Course: https://huggingface.co/learn
- Prompt Engineering Guide: https://www.promptingguide.ai

**Communities:**
- r/LocalLLaMA (Reddit)
- Hugging Face Discord
- OpenAI Developer Forum

---

*The best way to learn is by building. Start with something simple, experiment constantly, and don't be afraid to make mistakes. Every expert was once a beginner.*

---

**End of Lecture Notes**

