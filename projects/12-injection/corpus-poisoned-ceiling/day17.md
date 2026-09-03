# Day 17: AI Agents Course Overview & Multi-Model Orchestration - Lecture Notes

## Introduction

Welcome to Day 17! Today marks an important milestone in your AI journey as we explore the complete landscape of AI agents engineering. This session is primarily theoretical but incredibly important - it sets the foundation for everything you'll build in the coming weeks. Think of today as getting the map before starting your adventure.

## Part 1: Understanding the Course Structure

### The Three Pillars of Learning

This course is built on three fundamental pillars that work together to give you complete mastery:

**1. Theory - Understanding the "Why"**
Theory helps you understand what agents are and how they work. It's not just about copying code - it's about understanding the principles so you can design your own solutions. When you understand the theory, you can adapt to new frameworks and solve novel problems.

**2. Frameworks - Learning the "How"**
Frameworks are the tools that make building agents practical. Instead of reinventing the wheel, you'll learn to use powerful platforms that companies are using in production right now. Each framework has its strengths, and you'll learn when to use which one.

**3. Projects - Applying the "What"**
Projects bring everything together. You'll build real, commercial-grade applications that you could deploy tomorrow. These aren't toy examples - they're solutions to real business problems.

### The Six-Week Journey

The course is organized into six weeks, each building on the last. Let's understand what each week brings:

**Week 1: Foundations**
This is where you are now. Week 1 is about grounding yourself in the basics. You're learning what it means to have an "agentic architecture" - a system where multiple LLMs (Large Language Models) work together. The key project is a Career Q&A Agent - imagine having a digital version of yourself that can answer questions about your experience, skills, and career journey. You could put this on your website instead of a traditional resume!

**Week 2: OpenAI Agents SDK**
OpenAI's framework is elegant and simple. It's a great starting point because it's not overwhelming, but it's powerful enough to build real applications. You'll learn about guardrails - the safety mechanisms that keep your agents behaving properly. The Deep Research project will show you how to automate research tasks.

**Week 3: CrewAI**
CrewAI is the "fan favorite" according to the instructor. It's on the low-code end of the spectrum, meaning you can define your agents through configuration rather than writing lots of code. Think of it like assembling a crew of specialists who work together to solve problems.

**Week 4: LangGraph**
LangGraph is on the opposite end from CrewAI - it's full code and very sophisticated. It's complex but incredibly powerful. When you need maximum control and flexibility, LangGraph is your tool.

**Week 5: Autogen**
Microsoft's Autogen is special because it creates an environment where agents can collaborate remotely. It's like building a distributed team of AI agents that can work together even when they're running on different systems.

**Week 6: MCP and Capstone**
The Model Context Protocol (MCP) from Anthropic is brand new and exciting. It's an open-source way for different AI models to connect and share capabilities using a common protocol. Your capstone project will be a financial markets trading simulation where multiple agents make investment decisions based on real-time data.


## Part 2: What is an AI Agent?

### The Definition Challenge

Here's an interesting truth: there's no universally agreed-upon definition of what an "AI agent" is. The term has been hyped and used to mean many different things. But we need a working definition to move forward.

**HuggingFace's Definition:**
"AI agents are programs where LLM outputs control the workflow."

This is beautifully simple. It means that what the LLM decides determines what happens next. The AI is making decisions about the process flow, not just responding to fixed prompts.

### The Five Hallmarks of Agentic AI

In practice, people call something "agentic" if it has any of these five characteristics:

**1. Multiple LLM Calls**
When you're making several calls to language models in sequence, orchestrating them to solve a bigger problem, that's agentic. Even the simple exercise you did earlier - where one LLM generated a question and another answered it - could be called agentic.

**Example:** A system that first analyzes a document, then extracts key points, then generates a summary. Three LLM calls, each building on the last.

**2. Tool Use**
This is often considered the litmus test for agentic systems. When an LLM can use tools - like turning on lights, searching databases, or calling APIs - it's interacting with the real world. This is powerful because it means the AI isn't just generating text; it's taking actions.

**Example:** An agent that can search the web, read the results, decide if it needs more information, search again, and then answer your question based on what it found.

**3. Coordination Environment**
When you set up a system where different LLMs can send information to each other and coordinate their activities, that's agentic. It's like creating a workspace where multiple AI assistants can collaborate.

**Example:** A customer service system where one agent handles initial inquiries, another handles technical issues, and a third handles billing - all passing information between them.

**4. Planner LLM**
Having an LLM that coordinates and plans activities is a hallmark of agentic systems. This planner decides what needs to happen and in what order.

**Example:** A project management agent that breaks down a complex task, assigns subtasks to specialist agents, and monitors progress.

**5. Autonomy**
This is perhaps the most important hallmark. Autonomy means giving the LLM the ability to make decisions about what happens next. It can "choose its own adventure" in some sense.

**Example:** You ask an agent to plan a trip. It decides it needs to know your budget, asks you, then decides to search for flights, compares options, checks hotel availability, and presents a complete itinerary - all without you telling it each step.

### Understanding Autonomy

The word "autonomy" might sound scary - like we're creating AI that can do whatever it wants. But it's really about giving the AI decision-making power within boundaries. 

Think about it this way: When you asked an LLM to pick a business sector to analyze, you gave it autonomy to choose. It could have picked healthcare, finance, retail, or anything else. That choice then determined everything that followed. That's autonomy - the ability to make decisions that affect the workflow.

## Part 3: Workflows vs Agents

### Anthropic's Framework

Anthropic (the company behind Claude) wrote an excellent blog post called "Building Effective Agents." They distinguish between two types of agentic systems:

**Workflows:**
Systems where models and tools are orchestrated through predefined paths. You know the sequence of steps ahead of time. It's like following a recipe - you might have some flexibility, but the overall process is defined.

**Agents:**
Systems where models dynamically direct their own processes and tools. They maintain control over how tasks get accomplished. It's more like cooking without a recipe - you know what you want to make, but you decide the steps as you go based on what you observe.

This distinction is helpful, but notice that both fall under "agentic systems." So workflows are agentic, but they're more constrained than full agents. It's a spectrum, not a binary choice.

## Part 4: The Five Workflow Patterns

Let's dive deep into the five common workflow patterns. Understanding these will help you design better systems.

### Pattern 1: Prompt Chaining

**What It Is:**
Prompt chaining is the simplest pattern. You have a series of LLM calls where the output of one becomes the input to the next. Think of it like an assembly line - each station does one specific job, and the product moves down the line.

**How It Works:**
```
Input → LLM 1 → (optional code) → LLM 2 → (optional code) → LLM 3 → Output
```

Each LLM call is focused on one specific subtask. You might have some code in between to process or format the output before passing it to the next LLM.

**Why Use It:**
1. **Focused Prompts:** Each LLM call can have a perfectly crafted prompt for its specific task. This leads to better results.
2. **Guardrails:** The fixed sequence keeps everything on track. You know exactly what will happen.
3. **Debugging:** If something goes wrong, you know exactly which step failed.
4. **Optimization:** You can use different models for different steps - a cheap, fast model for simple tasks and a powerful model for complex ones.

**Real Example:**
Remember the exercise where you:
1. Asked an LLM to pick a business sector
2. Asked it to identify a pain point in that sector
3. Asked it to propose an AI solution

That's prompt chaining! Three LLM calls, each building on the last.

**Commercial Application:**
A content creation system might:
1. Generate article ideas (LLM 1)
2. Create an outline (LLM 2)
3. Write the full article (LLM 3)
4. Edit and polish (LLM 4)

Each step is specialized and optimized.

### Pattern 2: Routing

**What It Is:**
Routing is about intelligent task distribution. You have one LLM that acts as a router, deciding which specialist LLM should handle the task.

**How It Works:**
```
Input → Router LLM → Specialist LLM 1 (for task type A)
                   → Specialist LLM 2 (for task type B)
                   → Specialist LLM 3 (for task type C)
                   → Output
```

The router examines the input, classifies it, and sends it to the appropriate specialist.

**Why Use It:**
1. **Separation of Concerns:** Each specialist is expert in its domain.
2. **Efficiency:** You can use smaller, faster models for specific tasks instead of one giant model for everything.
3. **Cost Optimization:** Route simple tasks to cheap models, complex tasks to expensive ones.
4. **Scalability:** Easy to add new specialists without changing the routing logic much.

**Real Example:**
A customer service system:
- Router examines the customer's question
- Technical questions → Technical support agent
- Billing questions → Billing agent
- General questions → General support agent

**Commercial Application:**
An email processing system:
- Router reads incoming email
- Sales inquiries → Sales team agent
- Support requests → Support agent
- Complaints → Escalation agent

Each agent is trained/prompted specifically for its role.

**Note on Autonomy:**
Even though this is called a "workflow," notice that the router LLM has autonomy - it decides where to send the task. The line between workflows and agents is blurry!

### Pattern 3: Parallelization

**What It Is:**
Parallelization means breaking a task into multiple pieces that can be processed simultaneously, then combining the results.

**How It Works:**
```
Input → [Code splits task] → LLM 1 →
                            → LLM 2 → [Code combines results] → Output
                            → LLM 3 →
```

Notice that code (not an LLM) does the splitting and combining. The LLMs work in parallel on their pieces.

**Why Use It:**
1. **Speed:** Parallel processing is much faster than sequential.
2. **Scalability:** Can handle larger tasks by dividing them up.
3. **Multiple Perspectives:** Can get different viewpoints on the same problem.
4. **Redundancy:** Can run the same task multiple times and average results for better accuracy.

**Real Example:**
Analyzing a long document:
- Code splits document into chapters
- LLM 1 summarizes chapters 1-3
- LLM 2 summarizes chapters 4-6
- LLM 3 summarizes chapters 7-9
- Code combines summaries into final summary

**Another Example:**
Evaluating an idea:
- LLM 1 analyzes from business perspective
- LLM 2 analyzes from technical perspective
- LLM 3 analyzes from user perspective
- Code combines into comprehensive evaluation

**Commercial Application:**
Content moderation system:
- LLM 1 checks for hate speech
- LLM 2 checks for spam
- LLM 3 checks for misinformation
- All run simultaneously, code combines results to make final decision

**Interesting Variation:**
You could send the same task to three LLMs and take the majority vote or average their outputs. This improves reliability.

### Pattern 4: Orchestrator-Worker

**What It Is:**
This is like parallelization, but instead of code doing the splitting and combining, an LLM orchestrator does it. This makes the system much more dynamic and intelligent.

**How It Works:**
```
Input → Orchestrator LLM → Worker LLM 1 →
                         → Worker LLM 2 → Orchestrator LLM → Output
                         → Worker LLM 3 →
```

The orchestrator LLM decides how to break down the task, assigns work to workers, and synthesizes their results.

**Why Use It:**
1. **Dynamic Adaptation:** The orchestrator can adapt its strategy based on the specific task.
2. **Intelligent Distribution:** It can decide how many workers to use and what each should do.
3. **Smart Synthesis:** It understands the context when combining results, not just mechanically merging them.
4. **Flexibility:** Can handle varied and complex tasks that would be hard to code for.

**Real Example:**
Research task:
- Orchestrator receives: "Research the impact of AI on healthcare"
- Orchestrator decides to split into: current applications, future potential, ethical concerns, economic impact
- Assigns each topic to a worker LLM
- Workers research their topics
- Orchestrator synthesizes into coherent report

**Commercial Application:**
Product development:
- Orchestrator receives: "Design a new mobile app feature"
- Orchestrator assigns:
  - Worker 1: User experience considerations
  - Worker 2: Technical feasibility
  - Worker 3: Business value analysis
  - Worker 4: Competitive analysis
- Orchestrator combines insights into comprehensive feature specification

**Why This Feels Like an "Agent":**
Notice that the orchestrator has significant autonomy. It decides how to divide the work, how many workers to use, and how to combine results. Anthropic calls this a "workflow," but it has strong agent-like qualities. This shows how the boundaries between categories are fuzzy.

### Pattern 5: Evaluator-Optimizer (Validation Pattern)

**What It Is:**
This is the most commonly used pattern in production systems. One LLM generates content, and another LLM evaluates it. If the evaluation fails, the generator tries again with feedback.

**How It Works:**
```
Input → Generator LLM → Evaluator LLM → Accept → Output
              ↑                        ↓ Reject + Feedback
              └────────────────────────┘
```

It's a feedback loop that continues until the evaluator accepts the output or a maximum number of attempts is reached.

**Why Use It:**
1. **Quality Assurance:** Catches errors and problems before output reaches users.
2. **Improved Accuracy:** The feedback loop leads to better results.
3. **Robustness:** More reliable for production systems.
4. **Specialization:** Generator focuses on creation, evaluator focuses on validation.

**Real Example:**
Code generation:
- Generator LLM writes Python code
- Evaluator LLM checks:
  - Does it have syntax errors?
  - Does it follow best practices?
  - Does it handle edge cases?
  - Is it secure?
- If problems found: Evaluator explains issues, generator tries again
- If no problems: Code is approved

**Another Example:**
Content creation:
- Generator writes article
- Evaluator checks:
  - Is it factually accurate?
  - Is the tone appropriate?
  - Is it well-structured?
  - Does it meet length requirements?
- Feedback loop until quality standards met

**Commercial Application:**
Email response system:
- Generator drafts customer service email
- Evaluator checks:
  - Is it professional?
  - Does it answer the question?
  - Is it empathetic?
  - Does it follow company policies?
- Iterates until email meets standards

**Why This Matters:**
One of the biggest challenges with LLMs in production is reliability. They can make mistakes, hallucinate facts, or produce inconsistent results. The evaluator-optimizer pattern is your best defense against this. It's like having a quality control inspector on your assembly line.

**Advanced Variation:**
You can have multiple evaluators:
- Evaluator 1: Checks factual accuracy
- Evaluator 2: Checks tone and style
- Evaluator 3: Checks completeness

All must approve before output is accepted.

## Part 5: Agent Patterns (Open-Ended Systems)

### Moving Beyond Workflows

Now we shift from workflows (predefined paths) to true agent patterns (dynamic, open-ended systems).

**Key Differences:**
1. **No Fixed Path:** The agent decides its own sequence of actions.
2. **Feedback Loops:** Information flows back and forth continuously.
3. **Environment Interaction:** The agent interacts with an external environment.
4. **Open-Ended:** The process can continue indefinitely until the agent decides it's done.

**The Basic Agent Loop:**
```
Human Request → LLM → Action on Environment → Feedback from Environment → LLM → ...
                ↑                                                            ↓
                └────────────────────────────────────────────────────────────┘
```

The LLM continuously:
1. Observes the environment
2. Decides what action to take
3. Takes the action
4. Observes the results
5. Decides next action
6. Repeats until goal is achieved

### Understanding the Environment

The "environment" can be many things:
- **Web:** The agent can browse websites, click links, fill forms
- **Database:** The agent can query and update data
- **File System:** The agent can read and write files
- **APIs:** The agent can call external services
- **Physical World:** Through IoT devices, the agent can control lights, thermostats, etc.

### Why Agent Patterns Are Powerful

**Flexibility:**
The agent can adapt its strategy based on what it discovers. If one approach doesn't work, it tries another.

**Complex Problem Solving:**
Some problems are too complex to predefine all the steps. The agent figures out the path as it goes.

**Example:**
"Find me the best deal on a laptop with these specifications."

An agent might:
1. Search multiple shopping sites
2. Compare prices
3. Check reviews
4. Look for coupon codes
5. Compare shipping costs
6. Present the best option

You didn't tell it each step - it figured out what to do based on what it found.

### The Challenges of Agent Patterns

**Unpredictability:**
- You don't know how long it will take
- You don't know what path it will take
- You don't know exactly what the output will be
- You don't know how much it will cost (in API calls)

**Risk:**
- The agent might get stuck in a loop
- It might make unexpected decisions
- It might not complete the task at all
- It might take actions you didn't intend

**This is why monitoring and guardrails are essential!**

### Monitoring

Monitoring means having visibility into what your agents are doing. You need to see:
- What actions they're taking
- What decisions they're making
- How they're interacting with each other (in multi-agent systems)
- How much they're costing
- How long they're taking

**Tools for Monitoring:**
- **OpenAI SDK:** Built-in tracing capabilities
- **LangSmith:** Powerful monitoring for LangGraph systems
- **Custom Logging:** Your own tracking systems

**Why It Matters:**
Imagine deploying an agent system to production and it starts making thousands of API calls because it got stuck in a loop. Without monitoring, you wouldn't know until you got a huge bill. With monitoring, you catch it immediately.

### Guardrails

Guardrails are the safety mechanisms that keep agents behaving properly. They're like the bumpers in a bowling alley - they keep things on track.

**Types of Guardrails:**

**1. Input Guardrails:**
- Validate user requests before processing
- Block malicious or inappropriate inputs
- Ensure requests are within scope

**2. Process Guardrails:**
- Limit number of iterations
- Restrict which tools can be used
- Require approval for certain actions
- Set timeouts

**3. Output Guardrails:**
- Validate outputs before returning to user
- Check for sensitive information
- Ensure outputs meet quality standards

**Example Guardrails:**
```
- Maximum 10 iterations per task
- Cannot delete files without confirmation
- Cannot spend more than $10 in API calls per task
- Must get approval before sending emails
- Cannot access certain databases
```

**OpenAI SDK Quote:**
"Guardrails ensure agents behave safely, consistently, and within the boundaries you wish."

This is critical for production systems. You need confidence that your agents won't do something unexpected or harmful.

## Part 6: Working with Multiple LLMs

### The Model Landscape

Let's understand the different types of models available and when to use each.

**Closed Source (Proprietary) Models:**

**OpenAI:**
- **GPT-4:** The most capable, but expensive. Use for complex tasks requiring deep reasoning.
- **GPT-4-mini:** Cheaper, faster, still very capable. Great for most tasks.
- **O1/O3-mini:** Reasoning models that "think through" problems step by step. Excellent for math, logic, coding.

**Anthropic:**
- **Claude 3.7 Sonnet:** Excellent all-around model, great at following instructions, strong at coding.
- **Claude 3 Haiku:** Cheaper, faster version. Good for simpler tasks.

**Google:**
- **Gemini 2.0 Flash:** Currently has a free tier! Fast and capable. Great for experimentation without cost.
- **Gemini Pro:** More powerful version for complex tasks.

**DeepSeek:**
- **DeepSeek V3:** The full 671 billion parameter model. Very capable, very cheap.
- **DeepSeek R1:** Reasoning model, similar to OpenAI's O1.
- **Why DeepSeek Matters:** They achieved near-GPT-4 performance at a fraction of the training cost (30x less!). This is revolutionary for the industry.

**X.AI:**
- **Grok:** Elon Musk's model. Available through X platform.

**Open Source Models:**

**Llama (Meta):**
- **Llama 3.3 (70B):** Very powerful, but too large for most personal computers.
- **Llama 3.2 (3B):** Perfect for running locally. Good for simple tasks.
- **Llama 3.2 (1B):** Even smaller, very fast, but limited capability.

**Qwen (Alibaba):**
- Strong open-source alternative, various sizes available.

**Gemma (Google):**
- Google's open-source models, optimized for efficiency.

**Phi (Microsoft):**
- Small but capable models, good for edge devices.

**DeepSeek Distilled:**
- Smaller versions of DeepSeek trained on synthetic data.

### How to Choose a Model

**Consider These Factors:**

**1. Task Complexity:**
- Simple tasks (classification, basic Q&A): Use small, cheap models
- Complex tasks (reasoning, long-form generation): Use large, capable models

**2. Cost:**
- Development/testing: Use free or cheap models
- Production: Balance cost vs. quality

**3. Speed:**
- Real-time applications: Use fast models (even if less capable)
- Batch processing: Can use slower, more capable models

**4. Privacy:**
- Sensitive data: Use local models or private deployments
- Public data: Cloud models are fine

**5. Context Length:**
- Short inputs/outputs: Any model works
- Long documents: Need models with large context windows

### The Vellum Leaderboard

**URL:** https://www.vellum.ai/llm-leaderboard

This is an invaluable resource. It shows:
- **Side-by-side comparisons** of different models
- **Cost per million tokens** (input and output)
- **Context window sizes** (how much text they can process)
- **Performance benchmarks** across different tasks
- **Both open and closed source models**

**How to Use It:**
1. Identify your task type (coding, reasoning, general chat, etc.)
2. Check which models perform best for that task
3. Compare costs
4. Make informed decision

**Pro Tip:** Bookmark this and check it regularly. New models are released frequently, and prices change.

### Cost Realities

Let's be honest about costs because this concerns many students.

**Total Course Cost (Using Paid APIs):**
- Typically under $5 for the entire course
- Most exercises use very few tokens
- Even complex projects are cheap

**Initial Deposits:**
- OpenAI: $5 minimum (in US)
- Anthropic: $5 minimum
- DeepSeek: $2 minimum
- Groq: Pay-as-you-go, no minimum
- Gemini: Currently free tier available

**Free Options:**
- **Ollama:** Completely free, runs locally
- **Gemini:** Free tier (with usage limits)
- **DeepSeek:** So cheap it's almost free ($2 lasts a long time)

**The Reality of API Costs:**
Yes, it feels annoying to pay small amounts to different services. But consider:
- Running these models requires massive compute power
- Trillions of floating-point calculations per inference
- Huge electricity costs for providers
- Buying a computer powerful enough to run these models locally would cost thousands

**The value you get for a few dollars is actually incredible.**

**Strategy:**
- Use free/cheap models for learning and experimentation
- Use powerful models for final projects and demonstrations
- Mix and match based on task requirements

## Part 7: API Standardization

### The OpenAI API Standard

One of the best things that happened in the AI world is that OpenAI's API format became the de facto standard.

**Why This Matters:**
Instead of learning a different API for each model, you learn one format and use it everywhere.

**The Standard Format:**
```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="PROVIDER_ENDPOINT"  # Optional: defaults to OpenAI
)

response = client.chat.completions.create(
    model="model-name",
    messages=[
        {"role": "user", "content": "Your question here"}
    ]
)

answer = response.choices[0].message.content
```

**Who Uses This Format:**
- OpenAI (obviously)
- Google Gemini
- DeepSeek
- Groq
- Ollama (local)
- Many others

**The Exception:**
Anthropic has its own API format. It's similar but different:

```python
from anthropic import Anthropic

client = Anthropic(api_key="YOUR_API_KEY")

response = client.messages.create(
    model="claude-3-7-sonnet-latest",
    max_tokens=1024,  # Required for Anthropic
    messages=[
        {"role": "user", "content": "Your question here"}
    ]
)

answer = response.content[0].text
```

**Key Differences:**
- Different import and client class
- Requires `max_tokens` parameter
- Response structure slightly different

But the messages format is the same!

### Using OpenAI's Library for Other Providers

**For Google Gemini:**
```python
gemini = OpenAI(
    api_key=os.environ["GOOGLE_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
```

Notice the URL ends with "openai" - Google is saying "we have an endpoint that's compatible with OpenAI's format."

**For DeepSeek:**
```python
deepseek = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)
```

**For Groq:**
```python
groq = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)
```

Again, notice "openai" in the URL.

**For Ollama (Local):**
```python
ollama = OpenAI(
    api_key="ollama",  # Can be anything for local
    base_url="http://localhost:11434/v1"
)
```

**After Setup:**
All of them use the exact same `chat.completions.create()` call! This is incredibly convenient.

## Part 8: Running Models Locally with Ollama

### What is Ollama?

Ollama is a tool that lets you run open-source language models on your own computer. Think of it as creating your own private AI service.

**How It Works:**
1. Ollama runs as a service on your computer
2. It provides an API endpoint (localhost:11434)
3. This endpoint is compatible with OpenAI's API format
4. It uses highly optimized C++ code (llama.cpp) to run models efficiently

**Benefits:**
- **Free:** No API costs at all
- **Private:** Your data never leaves your computer
- **Offline:** Works without internet connection
- **Learning:** Great for experimentation

**Limitations:**
- **Model Size:** Can only run small models (1-8B parameters typically)
- **Speed:** Slower than cloud APIs (unless you have a very powerful computer)
- **Capability:** Smaller models are less capable than frontier models

### Installing Ollama

**Steps:**
1. Go to https://ollama.com
2. Click "Download"
3. Follow installation instructions for your operating system
4. Verify it's running by visiting http://localhost:11434

You should see: "Ollama is running"

**If It's Not Running:**
Open a terminal and type:
```bash
ollama serve
```

### Choosing Models for Ollama

**⚠️ CRITICAL WARNING:**
Do NOT try to run Llama 3.3 (70B) on a normal computer!

**Why:**
- It has 70 billion parameters
- Requires ~60-100GB of disk space
- Needs ~40GB of RAM to run
- Will overwhelm most computers

**Recommended Models:**

**For Good Balance:**
- **Llama 3.2 (3B):** 3 billion parameters, good capability, runs well on most computers
- **Qwen 2.5 (3B):** Similar size, strong performance

**For Maximum Speed:**
- **Llama 3.2 (1B):** 1 billion parameters, very fast, limited capability
- **Phi-3 (mini):** Microsoft's efficient small model

**For Best Local Performance:**
- **Llama 3.2 (7B):** If your computer can handle it
- **Gemma 2 (9B):** Google's model, good balance

**Specialized:**
- **DeepSeek-R1 distilled:** For reasoning tasks
- **CodeLlama:** Specialized for coding

### Using Ollama

**Download a Model:**
```bash
ollama pull llama3.2
```

This downloads the model to your computer.

**List Available Models:**
```bash
ollama list
```

**Run a Model Interactively:**
```bash
ollama run llama3.2
```

**Use in Python:**
```python
ollama = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

response = ollama.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Finding More Models

Visit https://ollama.com/library to browse all available models. You'll find:
- Different model families (Llama, Qwen, Gemma, Phi, etc.)
- Different sizes of each model
- Specialized models (coding, math, etc.)
- Community fine-tuned models

## Part 9: Practical Lab - Model Competition

### The Exercise

This lab demonstrates multiple workflow patterns in action. Let's break down what happened:

**Step 1: Question Generation**
```python
request = "Please come up with a challenging, nuanced question..."
messages = [{"role": "user", "content": request}]
response = openai.chat.completions.create(model="gpt-4-mini", messages=messages)
question = response.choices[0].message.content
```

GPT-4-mini generated: "How would you analyze the ethical implications of using AI in predictive policing, considering factors such as bias, accountability, and societal impact?"

**Step 2: Multiple Models Answer**
Six different models answered this question:
1. GPT-4-mini
2. Claude 3.7 Sonnet
3. Gemini 2.0 Flash
4. DeepSeek V3
5. Llama 3.3 (70B via Groq)
6. Llama 3.2 (3B via Ollama)

Each provided their analysis of AI in predictive policing.

**Step 3: Evaluation**
O3-mini (OpenAI's reasoning model) judged all responses and ranked them.

**Results:**
1. Gemini 2.0 Flash (winner!)
2. GPT-4-mini
3. Llama 3.3
4. DeepSeek
5. Claude 3.7 Sonnet
6. Llama 3.2

### Identifying the Patterns

**Which workflow patterns were used?**

**1. Prompt Chaining:**
The output of the first LLM (the question) became the input for the next LLMs (the answerers).

**2. Parallelization:**
Six models answered the question simultaneously. Their answers were collected and combined.

**3. Evaluator Pattern:**
O3-mini acted as the evaluator, assessing all the responses and ranking them.

**It's actually a hybrid of multiple patterns!**

### Key Insights from the Lab

**1. Model Differences Are Real:**
The same question produced very different responses. Some were comprehensive, others were more focused. Some included frameworks, others didn't.

**2. Size Isn't Everything:**
Gemini 2.0 Flash beat larger models. This shows that model architecture and training matter as much as size.

**3. Small Models Struggle:**
Llama 3.2 (3B) produced an incomplete response, showing the limitations of very small models for complex tasks.

**4. Evaluation Is Subjective:**
Different evaluators might rank differently. For more objectivity, you could:
- Have multiple evaluators vote
- Use specific criteria and score each
- Have humans validate the AI's rankings

### Python Techniques Demonstrated

**Zip Function:**
```python
for competitor, answer in zip(competitors, answers):
    print(f"{competitor}: {answer}")
```

This iterates through two lists simultaneously. Very useful for pairing related data.

**Enumerate Function:**
```python
for index, answer in enumerate(answers):
    print(f"Response {index + 1}: {answer}")
```

This gives you both the index and the item. Better than manually tracking a counter.

**F-strings with Literal Braces:**
```python
text = f"JSON format: {{'key': 'value'}}"
```

Use double braces `{{` and `}}` when you want actual braces in your f-string output.

**Triple-Quoted Strings:**
```python
text = """
This is a multi-line string.
No need for \n or string concatenation.
Very clean for long text blocks.
"""
```

Perfect for prompts and long text.

## Part 10: Development Environment

### Cursor IDE

**What It Is:**
Cursor is an AI-powered code editor built on VSCode. It uses LLMs to help you write code.

**Key Features:**
- **Code Completion:** AI suggests code as you type
- **Code Explanation:** Ask it to explain any code
- **Code Generation:** Describe what you want, it writes the code
- **Debugging Help:** It can help diagnose errors

**Why Use It:**
For this course, Cursor will dramatically speed up your development. It knows common patterns and can fill in boilerplate code automatically.

**Example:**
You type:
```python
response = openai.
```

Cursor suggests:
```python
response = openai.chat.completions.create(
    model="gpt-4-mini",
    messages=messages
)
```

It knows the pattern!

### UV Package Manager

**What It Is:**
UV is a modern Python package and environment manager. Think of it as a replacement for Anaconda or pip + venv.

**Why It's Better:**
- **Fast:** Written in Rust, incredibly quick
- **Simple:** Easy to use, less confusing than Anaconda
- **Modern:** Designed for current Python best practices
- **Integrated:** Many agent frameworks now use UV

**Basic Commands:**
```bash
# Create a new project
uv init myproject

# Add a package
uv add openai

# Run Python
uv run python script.py
```

**Why It Matters:**
Environment management is often a pain point for beginners. UV makes it much simpler and faster.

### Jupyter Notebooks in Cursor

**What They Are:**
Notebooks let you write code in "cells" that you can run independently. This is perfect for:
- **Learning:** Run code step by step, see results immediately
- **Experimentation:** Try different approaches easily
- **Documentation:** Mix code, results, and explanations

**How to Use:**
1. Open a `.ipynb` file in Cursor
2. Select your Python environment (the kernel)
3. Run cells with Shift+Enter
4. Add print statements to understand what's happening

**For Engineers:**
If you're used to traditional Python scripts, notebooks might feel weird at first. But they're valuable for:
- Research and development
- Prototyping
- Teaching and learning
- Data analysis

You'll use both notebooks and regular Python scripts in this course.

## Part 11: Commercial Applications

### Universal Applicability

The patterns you've learned today apply to virtually any AI application. Let's think about how:

**Content Generation:**
- **Prompt Chaining:** Idea → Outline → Draft → Edit → Final
- **Evaluator:** Quality check before publishing
- **Parallelization:** Generate multiple versions, pick best

**Customer Service:**
- **Routing:** Direct queries to appropriate specialist
- **Evaluator:** Ensure responses are appropriate and helpful
- **Orchestrator-Worker:** Break complex issues into subtasks

**Research and Analysis:**
- **Parallelization:** Analyze multiple sources simultaneously
- **Orchestrator-Worker:** Break research into subtopics
- **Prompt Chaining:** Gather data → Analyze → Synthesize → Report

**Software Development:**
- **Prompt Chaining:** Requirements → Design → Code → Test
- **Evaluator:** Code review and validation
- **Orchestrator-Worker:** Divide project into components

**Business Intelligence:**
- **Routing:** Different analyses for different data types
- **Parallelization:** Analyze multiple datasets simultaneously
- **Evaluator:** Validate insights before presenting

### Improving Robustness and Accuracy

**The Core Problem:**
Single LLM calls can be unreliable. They might:
- Make mistakes
- Miss important details
- Produce inconsistent results
- Hallucinate facts

**The Solution:**
Use workflow patterns to add layers of quality control:

**1. Multiple Attempts:**
Send the same request to multiple models, compare results.

**2. Evaluation Loops:**
Have one model check another's work.

**3. Decomposition:**
Break complex tasks into simpler subtasks.

**4. Specialization:**
Use different models for different parts of the task.

**Real-World Example:**
A legal document analysis system might:
1. **Routing:** Classify document type
2. **Parallelization:** Multiple models extract key information
3. **Orchestrator:** Synthesize findings
4. **Evaluator:** Legal expert model validates conclusions
5. **Human Review:** Final check before action

Each layer adds reliability.

## Part 12: Best Practices and Tips

### Learning Approach

**1. Watch First, Then Do:**
- Watch the instructor go through the lab
- Understand the concepts
- Then do it yourself
- Add your own experiments

**2. Experiment Freely:**
- Add print statements everywhere
- Change parameters and see what happens
- Try different models
- Break things and fix them

**3. Use AI to Learn:**
- Ask ChatGPT or Claude to explain concepts
- Use Cursor's AI to understand code
- Compare answers from different models
- Have one model evaluate another's explanation

**4. Debug Thoughtfully:**
- Read error messages carefully
- Use print statements to understand flow
- Check the troubleshooting guides
- Ask for help when stuck

### Debugging Tips

**Common Issues:**

**Name Errors:**
Usually means you forgot to run a cell that defines a variable. Run cells in order!

**Import Errors:**
Your environment might not have the package. Check that you selected the right kernel.

**API Errors:**
- Check your API key is set correctly
- Verify you have credit/quota
- Check the model name is correct

**Unexpected Results:**
- Print intermediate values
- Check your prompt carefully
- Try a different model
- Simplify the task

### Community Engagement

**1. Share Your Work:**
- Create a GitHub repository
- Post projects on LinkedIn
- Tag the instructor for amplification
- Help other students

**2. Contribute:**
- Submit pull requests with improvements
- Share interesting variations
- Document problems and solutions
- Add to community examples

**3. Network:**
- Connect with other students
- Join discussions
- Share resources
- Collaborate on projects

**Why This Matters:**
Building in public:
- Demonstrates your skills to potential employers
- Gets you feedback and ideas
- Builds your professional network
- Contributes to the community

## Part 13: Looking Ahead

### What's Next

**Day 4: Tool Use**
You'll learn how to give LLMs the ability to use tools - calling APIs, searching databases, interacting with external systems. This is where agents become truly powerful.

**Day 5: Career Q&A Agent**
Your first major project! Build an agent that can answer questions about your career and experience. Deploy it to the web!

**Week 2 and Beyond:**
Each week introduces a new framework and new projects, building on what you've learned.

### Key Concepts to Remember

**1. Workflows vs Agents:**
Workflows have predefined paths, agents are more dynamic. Both are valuable.

**2. The Five Workflow Patterns:**
- Prompt Chaining
- Routing
- Parallelization
- Orchestrator-Worker
- Evaluator-Optimizer

**3. Autonomy:**
Giving LLMs decision-making power within boundaries.

**4. Monitoring and Guardrails:**
Essential for production systems.

**5. Model Selection:**
Choose based on task complexity, cost, speed, and privacy needs.

### Exercises to Try

**1. Add a Pattern:**
Take the model competition lab and add another workflow pattern. Maybe:
- Have multiple evaluators vote
- Route different question types to different models
- Chain the results through additional analysis

**2. Experiment with Models:**
Try the same task with different models. Compare:
- Quality of results
- Speed
- Cost
- Consistency

**3. Build Something New:**
Apply these patterns to a problem you care about:
- Automate a task at work
- Create a personal assistant
- Build a research tool
- Make a content generator

**4. Improve Reliability:**
Take a single LLM call and add:
- An evaluator to check the output
- Multiple attempts with voting
- Decomposition into subtasks

## Conclusion

Today you've learned the theoretical foundation of AI agents. You understand:
- What makes something "agentic"
- The difference between workflows and agents
- Five powerful workflow patterns
- How to work with multiple LLMs
- The tools and environment for development

This knowledge is the foundation for everything that follows. As you build more projects, these patterns will become second nature. You'll start seeing opportunities to apply them everywhere.

**Remember:**
- Theory guides practice
- Patterns solve problems
- Experimentation leads to learning
- Community makes you stronger

**The journey to mastering AI agents has truly begun!**

## Additional Resources

### Essential Links

1. **Anthropic's "Building Effective Agents":**
   Search for this blog post - it's the source of the workflow patterns and excellent reading.

2. **Vellum Leaderboard:**
   https://www.vellum.ai/llm-leaderboard
   Bookmark this for model comparisons.

3. **Ollama:**
   https://ollama.com
   For running models locally.

4. **Course Repository:**
   Check your course materials for the GitHub link with all labs and guides.

### Further Reading

**Understanding LLMs:**
- How transformers work
- Attention mechanisms
- Training vs inference
- Prompt engineering

**Agent Frameworks:**
- OpenAI Agents SDK documentation
- CrewAI documentation
- LangGraph tutorials
- Autogen examples

**Production Considerations:**
- Monitoring and observability
- Cost optimization
- Latency reduction
- Security and privacy

### Community Resources

**GitHub:**
- Submit pull requests
- Browse community contributions
- Share your projects

**LinkedIn:**
- Connect with instructor
- Share your progress
- Network with other students

**Course Forums:**
- Ask questions
- Help others
- Share insights

---

**Final Thought:**

AI agents represent a fundamental shift in how we build software. Instead of programming every step, we're orchestrating intelligent systems that can adapt, learn, and solve complex problems. The patterns you've learned today are your toolkit for this new paradigm.

Keep building, keep learning, and most importantly, keep experimenting. The best way to understand agents is to build them!

**Welcome to the future of AI engineering!**

