# Day 19: Asynchronous Python & OpenAI Agents SDK - Lecture Notes

## Welcome to Day 19: OpenAI Agents SDK

Welcome to an exciting new phase of our journey into AI agents! Today marks the beginning of Day 19, where we'll explore the OpenAI Agents SDK, one of the newest and most elegant agent frameworks available. This framework was previously known as Swarm, and it represents OpenAI's approach to making agent development both powerful and accessible.

But before we dive into the OpenAI Agents SDK, we have an important foundation to lay: understanding asynchronous Python with async IO. This might seem like a detour, but trust me—it's absolutely essential for working with any modern agent framework.

---

## Part 1: Understanding Asynchronous Python (Async IO)

### Why Async IO Matters for AI Agents

Every major agent framework you'll encounter—whether it's OpenAI's Agents SDK, LangGraph, CrewAI, or others—uses asynchronous Python. This isn't a coincidence; it's because async IO solves a fundamental problem in agent development: how to efficiently manage multiple AI operations that spend most of their time waiting for API responses.

Think about what happens when you call an LLM API like OpenAI's GPT-4:
1. Your code sends a request over the network
2. Your code waits... and waits... and waits...
3. The API processes your request
4. A response comes back over the network
5. Your code continues

During that waiting period (steps 2-4), your program is just sitting idle. If you have multiple agents or multiple LLM calls to make, wouldn't it be great if you could make other progress while waiting? That's exactly what async IO enables.

### The Short Version: Just Two Keywords

If you're in a hurry and want to get by with minimal understanding, here's the simplest possible explanation:

**Keyword 1: `async`**
- Put this before `def` when defining a function
- Example: `async def my_function():`
- This makes it an "async function" or "coroutine"

**Keyword 2: `await`**
- Put this before calling an async function
- Example: `result = await my_function()`
- This actually runs the coroutine and waits for it to finish

That's it! You could follow just these two rules and get your code to work. But let's go deeper and really understand what's happening.

### The Real Story: How Async IO Actually Works

Async IO is Python's lightweight alternative to traditional multithreading or multiprocessing. It was introduced in Python 3.5 and has become increasingly important in modern Python development.

**Traditional Approaches to Concurrency:**

1. **Multithreading:** Your program runs multiple threads, and the operating system rapidly switches between them, making it seem like they're running simultaneously. This is powerful but comes with complexity—managing thread safety, locks, race conditions, etc.

2. **Multiprocessing:** Your program spawns multiple Python processes, each with its own memory space. This is great for CPU-intensive tasks but is heavy on resources.

**Async IO's Approach:**

Async IO provides concurrency without using OS-level threads or multiple processes. Instead, it uses:
- Special Python keywords (`async` and `await`)
- An event loop (more on this soon)
- Cooperative multitasking (coroutines voluntarily yield control)

Because it's implemented at the Python code level rather than the OS level, it's extremely lightweight. You can have thousands or even tens of thousands of concurrent operations with minimal resource overhead.

### Understanding Coroutines vs Functions

This is where many beginners get confused, so let's be very clear about the distinction:

**Regular Function:**
```python
def regular_function():
    print("I'm running!")
    return "done"

result = regular_function()  # This immediately prints "I'm running!" and returns "done"
```

**Coroutine (Async Function):**
```python
async def my_coroutine():
    print("I'm running!")
    return "done"

coroutine_obj = my_coroutine()  # This does NOT print anything!
# coroutine_obj is a coroutine object, not "done"
```

When you call a regular function, it executes immediately. When you call a coroutine, it does NOT execute—it just returns a coroutine object. To actually run a coroutine, you need to **schedule it for execution**, typically using `await`:

```python
async def my_coroutine():
    print("I'm running!")
    return "done"

result = await my_coroutine()  # NOW it prints "I'm running!" and result is "done"
```

### The Event Loop: The Heart of Async IO

So what happens when you use `await`? Behind the scenes, there's something called the **event loop** managing everything.

The event loop is essentially a sophisticated while loop that:
1. Maintains a queue of coroutines waiting to be executed
2. Executes one coroutine at a time (it's not true parallel processing)
3. When a coroutine hits a point where it's waiting (like for an API response), the event loop pauses it and switches to another coroutine
4. When the waiting is done (API response arrives), the event loop resumes the paused coroutine

Here's a simple mental model:

```
Event Loop's Job:
- Run Coroutine A until it needs to wait
- Switch to Coroutine B, run until it needs to wait
- Switch to Coroutine C, run until it needs to wait
- Check if Coroutine A's wait is done → if yes, resume it
- Check if Coroutine B's wait is done → if yes, resume it
- And so on...
```

This is sometimes called "cooperative multitasking" because coroutines cooperate by voluntarily yielding control when they're waiting.

### Practical Example: Understanding await

Let's see a complete example to make this concrete:

```python
import asyncio

async def fetch_data(id):
    print(f"Starting to fetch data for {id}")
    await asyncio.sleep(2)  # Simulates waiting for an API call
    print(f"Finished fetching data for {id}")
    return f"Data for {id}"

# This won't work (outside async context):
# result = await fetch_data(1)

# Need to run it properly:
async def main():
    result = await fetch_data(1)
    print(f"Got result: {result}")

# Run the async function
asyncio.run(main())
```

Output:
```
Starting to fetch data for 1
(2 second pause)
Finished fetching data for 1
Got result: Data for 1
```

### Running Multiple Coroutines Concurrently

The real power of async IO comes from running multiple operations concurrently. The most common way to do this is with `asyncio.gather()`:

```python
import asyncio

async def fetch_data(id):
    print(f"Starting to fetch data for {id}")
    await asyncio.sleep(2)
    print(f"Finished fetching data for {id}")
    return f"Data for {id}"

async def main():
    # Run three fetch operations concurrently
    results = await asyncio.gather(
        fetch_data(1),
        fetch_data(2),
        fetch_data(3)
    )
    print(f"All results: {results}")

asyncio.run(main())
```

Output:
```
Starting to fetch data for 1
Starting to fetch data for 2
Starting to fetch data for 3
(2 second pause)
Finished fetching data for 1
Finished fetching data for 2
Finished fetching data for 3
All results: ['Data for 1', 'Data for 2', 'Data for 3']
```

Notice how:
1. All three start almost simultaneously
2. They all wait together during the 2-second sleep
3. They all finish around the same time
4. Total time is ~2 seconds, not 6 seconds!

If we had called them sequentially (`await fetch_data(1)`, then `await fetch_data(2)`, then `await fetch_data(3)`), it would take 6 seconds total. With `asyncio.gather()`, we get 3x speedup!

### Why Every Agent Framework Uses Async IO

Now it should be clear why async IO is perfect for agent frameworks:

1. **Multiple LLM Calls:** Agents often need to call LLMs multiple times. Each call involves network latency and processing time. With async, you can make multiple calls concurrently.

2. **Multi-Agent Systems:** When you have multiple agents working together, they can operate concurrently without blocking each other.

3. **Efficient Resource Usage:** Async IO allows thousands of concurrent operations with minimal memory overhead, perfect for scalable agent systems.

4. **I/O-Bound Operations:** Agent work is mostly I/O-bound (waiting for API responses), not CPU-bound, making async IO ideal.

### Key Takeaways on Async IO

Before we move on, let's summarize the essential points:

- **`async def`** creates a coroutine, not a regular function
- Calling a coroutine returns a coroutine object but doesn't execute it
- **`await`** schedules a coroutine for execution and waits for its result
- The **event loop** manages execution, switching between coroutines when they're waiting
- **`asyncio.gather()`** runs multiple coroutines concurrently
- Async IO is lightweight, efficient, and perfect for I/O-bound operations like API calls

---

## Part 2: Introduction to OpenAI Agents SDK

Now that we understand async IO, we can dive into the OpenAI Agents SDK with full comprehension of how it works under the hood.

### What is OpenAI Agents SDK?

The OpenAI Agents SDK (previously called Swarm) is a Python framework for building multi-agent systems. It was released very recently and represents OpenAI's vision for how agent development should work.

**Key Characteristics:**

1. **Lightweight:** Minimal abstraction layers, stays close to "bare metal" Python
2. **Flexible:** Not opinionated about how you structure your agents
3. **Convenient:** Handles boilerplate (tool definitions, JSON schemas) automatically
4. **Powerful:** Supports complex multi-agent architectures

### The Philosophy: Unopinionated but Helpful

Some frameworks are very "opinionated"—they prescribe specific ways of doing things, specific patterns you must follow, specific architecture you must adopt. This can be good because it guides you toward best practices and makes certain tasks very quick.

The OpenAI Agents SDK takes a different approach: it's relatively unopinionated, giving you flexibility to structure your system however makes sense for your use case. But it's still helpful where it matters—it takes away the tedious boilerplate work (like writing JSON schemas for tools) while leaving you in full control of the logic.

### Core Concepts and Terminology

The OpenAI Agents SDK introduces three main concepts:

**1. Agent**

An agent is a package around LLM calls that has:
- A **name** (identifier)
- **Instructions** (essentially a system prompt)
- A **model** (which LLM to use)
- Optional **tools** (functions/capabilities it can use)
- Optional **handoffs** (other agents it can delegate to)

Think of an agent as a specialized AI assistant focused on one particular role or task.

**2. Handoffs**

A handoff is when one agent delegates control to another agent. The key characteristic: **control doesn't return to the first agent**. It's like handing off a baton in a relay race—once you've handed it off, your part is done.

Example: A sales manager agent might generate three email drafts, pick the best one, then hand off to an email formatting agent to convert it to HTML and send it.

**3. Guardrails**

Guardrails are checks and controls you put in place to ensure agents behave as intended. These can include:
- Input validation (checking what goes into the agent)
- Output validation (checking what comes out)
- Behavioral constraints (limiting what actions an agent can take)

The term "guardrails" is common in software engineering and refers to protective measures that prevent the system from doing something unwanted.

### The Three Steps to Run an Agent

Every time you want to execute an agent, you follow this pattern:

**Step 1: Create an Agent Instance**
```python
agent = Agent(
    name="my_agent",
    instructions="You are a helpful assistant",
    model="gpt-4o-mini"
)
```

**Step 2: Wrap in a Trace (Optional but Recommended)**
```python
with trace("my_task"):
    # agent execution goes here
```

The trace context manager logs all interactions and makes them available in OpenAI's monitoring dashboard at platform.openai.com/traces.

**Step 3: Call runner.run**
```python
result = await runner.run(agent, "your message here")
```

Remember, `runner.run` is an async function (coroutine), so you must use `await`.

### Your First Agent: A Complete Example

Let's build a simple joke-telling agent:

```python
from dotenv import load_dotenv
from agents import Agent, runner, trace

# Load environment variables (OPENAI_API_KEY)
load_dotenv(override=True)

# Create the agent
jokester = Agent(
    name="jokester",
    instructions="You are a joke teller.",
    model="gpt-4o-mini"
)

# Run the agent with tracing
async def tell_joke():
    with trace("telling_a_joke"):
        result = await runner.run(
            jokester,
            "Tell a joke about autonomous AI agents"
        )
        print(result.final_output)

# Execute
await tell_joke()
```

Output might be:
```
Why don't autonomous agents ever get lost?
Because they are always following their own self-driving instructions!
```

### Understanding the Package Structure

The OpenAI Agents SDK is part of the `openai` Python package, in a module called `agents`. You import from it like this:

```python
from agents import Agent, runner, trace, function_tool
```

Note: The use of the generic name "agents" can cause conflicts if you have your own module named "agents", but that's what OpenAI chose.

### Viewing Traces in OpenAI Platform

After running your agent with the trace context manager, you can view detailed logs:

1. Go to platform.openai.com
2. Click "Traces" in the left sidebar
3. Find your trace (named what you put in `with trace("name")`)
4. Click to expand and see:
   - All LLM calls made
   - System prompts used
   - User messages
   - Assistant responses
   - Tool calls (if any)
   - Timing information
   - Token usage

This is invaluable for debugging and understanding complex multi-agent workflows.

---

## Part 3: Streaming Responses

Sometimes you want to see the agent's response as it's being generated, rather than waiting for the complete response. This is called streaming.

### Using runner.run_streamed

Instead of `runner.run()`, use `runner.run_streamed()`:

```python
async def stream_joke():
    # Note: no await here!
    stream = runner.run_streamed(
        jokester,
        "Tell a joke about AI"
    )
    
    # Use async for to iterate
    async for chunk in stream:
        # Check if this chunk contains text
        if hasattr(chunk, 'text') and chunk.text:
            print(chunk.text, end='', flush=True)
    
    print()  # New line at the end

await stream_joke()
```

The pattern is:
1. Call `run_streamed()` **without await** to get a stream object
2. Use `async for` to iterate through chunks as they arrive
3. Process each chunk (usually just print it)

This provides a much better user experience for long responses, as users see progress immediately rather than waiting for everything to complete.

---

## Part 4: Vibe Coding - Best Practices for LLM-Assisted Development

Before we get deeper into agent development, let's talk about a modern coding practice that's incredibly relevant: **vibe coding**.

### What is Vibe Coding?

The term was coined by Andrej Karpathy (legendary AI researcher, former Tesla AI director) to describe the experience of coding with LLM assistance. It's this flow state where you:
- Ask an LLM for code
- Tweak it a bit
- Ask for more
- Iterate quickly
- Make impressive progress fast

It's called "vibe" coding because you're kind of going with the flow, letting the LLM handle boilerplate while you focus on the bigger picture.

### Five Essential Tips for Effective Vibe Coding

**Tip 1: Good Vibes - Craft Quality Prompts**

Spend time creating reusable prompts that work well. Good practices:
- Ask for **concise, clean code** (LLMs tend toward verbose solutions)
- Mention **today's date** (e.g., "As of January 2026, what's the best way to...")
- This helps LLM use current APIs rather than outdated ones
- Request minimal exception handling for prototyping (you can add it later)

Example:
"As of January 2026, write concise Python code to call the OpenAI API. Use the latest API syntax. Keep it under 20 lines."

**Tip 2: Vibe but Verify - Cross-Reference Multiple LLMs**

Don't rely on a single LLM. Ask the same question to:
- ChatGPT (GPT-4)
- Claude (Anthropic)
- Maybe even Gemini or DeepSeek

Why? Often one will give a much better answer than the other. Sometimes one will catch bugs or issues the other missed. You learn from comparing different approaches.

**Tip 3: Step Up the Vibe - Work in Small Increments**

This is the most important tip. **Never generate 200 lines of code at once.**

Here's what happens when you do:
- The code will have multiple bugs
- You won't understand it all
- Debugging becomes a nightmare
- You get stuck

Better approach:
- Break the problem into small steps (10-15 lines each)
- Generate one function at a time
- Test each piece independently
- Build up to the complete solution

**Advanced technique:** If you don't know how to break down the problem, ask an LLM:
"I need to [describe problem]. Please suggest 4-5 simple steps to solve this, where each step is independently testable. Don't write code yet, just describe the steps."

Then generate code for each step one at a time.

**Tip 4: Vibe and Validate - Use LLMs to Review Code**

This mirrors the evaluator-optimizer pattern:
1. Ask LLM #1 to generate code
2. Take that code to LLM #2 and say: "Please review this code. Are there any bugs? Can it be improved or made more concise?"
3. Incorporate suggested improvements

This often catches issues and leads to cleaner solutions.

**Tip 5: Vibe with Variety - Request Multiple Approaches**

Instead of: "Write code to do X"
Try: "Show me three different ways to do X in Python"

This forces the LLM to think more deeply about the problem and consider alternatives. You'll often get:
- A simple approach
- An optimized approach
- A more modern/elegant approach

Then you can choose the best one or combine ideas.

### The Golden Rule: Always Understand the Code

While vibe coding is fun and productive, **never** use code you don't understand. If an LLM generates something unclear:
- Ask it to explain the code line by line
- Ask "Why did you choose this approach?"
- Ask "What are the alternatives?"

Understanding is essential because when (not if) something breaks, you need to be able to debug it.

---

## Part 5: Building an Agent Workflow - The Sales Development Rep

Now let's apply everything we've learned to build a practical multi-agent system. We'll create an automated Sales Development Representative (SDR) that can write and send cold sales emails.

### Project Overview: Three Layers of Architecture

We'll build this in three progressive layers:

**Layer 1: Simple Workflow**
- Multiple agents, manually orchestrated in Python
- Sequential and parallel execution
- Basic agent coordination

**Layer 2: Agents with Tools**
- Agents that can call Python functions
- Simplified tool definition with decorators
- External API integration (SendGrid for email)

**Layer 3: Multi-Agent Collaboration**
- Agents calling other agents
- Tools vs. handoffs
- Complex workflows with delegation

### Layer 1: Manual Agent Workflow

Let's start simple. We'll create three sales agents, each with a different writing style, and have them all generate cold emails. Then we'll use a fourth agent to pick the best one.

**Step 1: Create Three Sales Agents**

```python
# Instructions for three different styles
professional_instructions = """
You are a sales agent working for CompliAI, a SaaS tool that 
helps companies prepare for SOC 2 compliance audits.
You write professional, serious cold emails.
"""

engaging_instructions = """
You are a humorous, engaging sales agent working for CompliAI.
You write witty, engaging cold emails that are likely to get a response.
"""

concise_instructions = """
You are a busy sales agent. You write concise, to-the-point cold emails.
"""

# Create the agents
sales_agent_1 = Agent(
    name="professional_sales_agent",
    instructions=professional_instructions,
    model="gpt-4o-mini"
)

sales_agent_2 = Agent(
    name="engaging_sales_agent",
    instructions=engaging_instructions,
    model="gpt-4o-mini"
)

sales_agent_3 = Agent(
    name="concise_sales_agent",
    instructions=concise_instructions,
    model="gpt-4o-mini"
)
```

**Step 2: Run All Three in Parallel**

```python
import asyncio

async def generate_three_emails():
    message = "Write a cold sales email"
    
    # Run all three agents concurrently
    results = await asyncio.gather(
        runner.run(sales_agent_1, message),
        runner.run(sales_agent_2, message),
        runner.run(sales_agent_3, message)
    )
    
    # Extract the text from each result
    emails = [result.final_output for result in results]
    
    return emails

emails = await generate_three_emails()
for i, email in enumerate(emails, 1):
    print(f"\n=== Email {i} ===\n{email}")
```

**Step 3: Create a Picker Agent**

```python
sales_picker = Agent(
    name="sales_picker",
    instructions="""
    Pick the best cold sales email from the given options.
    Imagine you are a customer. Pick the one you're most likely 
    to respond to. Don't give an explanation. Reply with the email 
    you select only.
    """,
    model="gpt-4o-mini"
)

async def pick_best_email(emails):
    # Combine emails into one message
    email_options = "\n\n---\n\n".join(emails)
    message = f"Here are three email options:\n\n{email_options}\n\nPick the best one."
    
    result = await runner.run(sales_picker, message)
    return result.final_output

best_email = await pick_best_email(emails)
print(f"\n=== BEST EMAIL ===\n{best_email}")
```

**What Pattern Is This?**

This is a variation of the **evaluator-optimizer** pattern from Anthropic's design patterns. We're:
1. Generating multiple options (parallelization pattern)
2. Evaluating them with another LLM to pick the best (evaluator pattern)

### Layer 2: Adding Tools

Now let's make our agents more powerful by giving them tools. First, let's understand how tools work in the OpenAI Agents SDK.

**The Old Way (Day 18):**
- Write JSON schemas by hand
- Create tool definitions with parameters
- Write handler functions with if statements
- Lots of boilerplate code

**The New Way (OpenAI Agents SDK):**
- Write a normal Python function
- Add `@function_tool` decorator
- Done! SDK auto-generates everything

**Example: Creating a Tool from a Function**

```python
from agents import function_tool
import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

@function_tool
def send_email(email_body: str):
    """
    Send out an email with the given body to all sales prospects.
    """
    from_email = "your-verified@email.com"  # Must be verified in SendGrid
    to_email = "recipient@email.com"
    
    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject="About CompliAI",
        plain_text_content=email_body
    )
    
    sg = SendGridAPIClient(os.getenv('SENDGRID_API_KEY'))
    response = sg.send(message)
    
    return "Email sent successfully"
```

What happened here?
1. We wrote a normal Python function with type hints
2. We added a docstring (becomes tool description)
3. We decorated with `@function_tool`
4. The SDK automatically created:
   - Tool name ("send_email")
   - Tool description (from docstring)
   - JSON parameter schema (from type hints)

**Setting Up SendGrid:**

To actually send emails, you need to:
1. Go to sendgrid.com and sign up (free)
2. Create an API key (Settings → API Keys)
3. Verify a sender email address (Settings → Sender Authentication)
4. Add to your .env file: `SENDGRID_API_KEY=your_key_here`

### Agents as Tools: A Powerful Concept

Here's where it gets interesting: **you can wrap an agent to make it into a tool**.

```python
# Wrap sales_agent_1 as a tool
tool_1 = sales_agent_1.as_tool(
    tool_name="sales_agent_1",
    description="Write a cold sales email"
)
```

Now `tool_1` is a FunctionTool that, when called, will actually run `sales_agent_1`. This lets you:
- Package agents as reusable tools
- Give agents to other agents as capabilities
- Build hierarchical agent systems

**Creating Multiple Agent Tools:**

```python
tool_1 = sales_agent_1.as_tool(
    tool_name="sales_agent_1",
    description="Write a cold sales email"
)

tool_2 = sales_agent_2.as_tool(
    tool_name="sales_agent_2",
    description="Write a cold sales email"
)

tool_3 = sales_agent_3.as_tool(
    tool_name="sales_agent_3",
    description="Write a cold sales email"
)

# Combine with the send_email function tool
tools = [tool_1, tool_2, tool_3, send_email]
```

### The Sales Manager Agent

Now we create a "manager" agent that can use all these tools:

```python
sales_manager = Agent(
    name="sales_manager",
    instructions="""
    You are a sales manager working for CompliAI.
    You use the tools given to you to generate cold sales emails.
    You never generate sales emails yourself. You always use the tools.
    You try all three tools once before choosing the best.
    You pick the single best email and use the send_email tool 
    to send the best email and only the best email to the user.
    """,
    tools=tools,
    model="gpt-4o-mini"
)

# Run it
async def automated_sales():
    with trace("sales_manager"):
        result = await runner.run(
            sales_manager,
            "Send a cold sales email addressed to 'dear CEO'"
        )
        print(result.final_output)

await automated_sales()
```

**What's Happening:**
1. Sales manager is given four tools (three agents + send function)
2. It autonomously decides to call all three sales agent tools
3. It evaluates the results
4. It picks the best one
5. It calls the send_email tool to actually send it

This is now becoming truly "agentic"—the agent is making decisions about which tools to use and when.

### Layer 3: Handoffs

We have one more concept to introduce: **handoffs**. These are similar to agent-as-tool, but with a key difference:

**Tools (Request-Response Pattern):**
- Agent calls tool
- Tool executes and returns result
- Control returns to calling agent
- Agent continues with the result

**Handoffs (Delegation Pattern):**
- Agent delegates to another agent
- Control passes completely
- Original agent is done, doesn't get control back

**When to Use Each:**
- **Tools:** When an agent needs a capability to help with its task
- **Handoffs:** When an agent has finished its part and another agent should take over

**Example: Email Formatter as Handoff**

Let's create specialized agents for formatting and sending:

```python
# Agent to write email subjects
subject_writer = Agent(
    name="subject_writer",
    instructions="You write compelling subject lines for emails.",
    model="gpt-4o-mini"
)

# Agent to convert to HTML
html_converter = Agent(
    name="html_converter",
    instructions="""
    You convert text emails to HTML with simple, clear, 
    compelling layout and design.
    """,
    model="gpt-4o-mini"
)

# Wrap them as tools
subject_tool = subject_writer.as_tool(
    tool_name="subject_writer",
    description="Write a subject for an email"
)

html_tool = html_converter.as_tool(
    tool_name="html_converter",
    description="Convert text email to HTML"
)

# Function to send HTML email
@function_tool
def send_html_email(subject: str, body: str):
    """
    Send out an email with the given subject and HTML body 
    to all sales prospects.
    """
    # Similar to send_email but sends HTML
    # ... implementation details ...
    return "HTML email sent successfully"

# Create the email manager agent
emailer = Agent(
    name="email_manager",
    instructions="""
    You are an email formatter and sender.
    You receive the body of an email.
    You first use the subject_writer tool, then the html_converter tool,
    and finally you send the email.
    """,
    tools=[subject_tool, html_tool, send_html_email],
    model="gpt-4o-mini",
    handoff_description="Convert an email to HTML and send it"
)
```

**The handoff_description** is how this agent advertises itself to other agents that might want to delegate to it.

**Creating the Final Sales Manager with Handoff:**

```python
# Three sales agent tools (same as before)
sales_tools = [tool_1, tool_2, tool_3]

# Sales manager now has both tools AND a handoff
final_sales_manager = Agent(
    name="sales_manager",
    instructions="""
    You are a sales manager working for CompliAI.
    You use the tools given to you to generate cold sales emails.
    Never generate one yourself. Always use the tools.
    You try all three sales email tools at least once before choosing the best one.
    You can use all the tools multiple times if you're not satisfied with the results.
    You select the single best email using your own judgment of which 
    email will be most effective, and then after picking it, you hand off 
    to the email_manager agent to format and send the email.
    """,
    tools=sales_tools,
    handoffs=[emailer],
    model="gpt-4o-mini"
)

# Run the complete system
async def complete_sdr():
    with trace("automated_sdr"):
        result = await runner.run(
            final_sales_manager,
            "Create and send a sales email"
        )
        print(result.final_output)

await complete_sdr()
```

**The Complete Flow:**
1. Sales manager calls three sales agent tools
2. Sales manager evaluates results and picks best
3. Sales manager **hands off** to email_manager
4. Email manager calls subject_writer tool
5. Email manager calls html_converter tool
6. Email manager calls send_html_email tool
7. Done! Email is sent

You can see all of this beautifully visualized in the OpenAI traces dashboard.

---

## Part 6: Understanding the Difference: Tools vs Handoffs

This is subtle but important, so let's be very clear:

**Tools:**
- Think of them as capabilities or features
- The agent uses them to accomplish part of its task
- Control flow: Agent → Tool → Back to Agent → Continue
- Example: A writer using a spell-checker

**Handoffs:**
- Think of them as delegation or passing the baton
- The agent is done with its part and another takes over
- Control flow: Agent A → Agent B (A is finished)
- Example: A sales rep passing a qualified lead to account management

**Technical Difference:**
- Tools return results to the calling agent
- Handoffs transfer control permanently (in that execution)

**When to Use:**
- Tools: When the agent needs help with a subtask but remains in charge
- Handoffs: When the agent's role is complete and a different role should take over

---

## Part 7: Commercial Applications

Let's think about how this applies to real businesses:

**Sales Automation:**
- Generate personalized cold emails at scale
- Follow-up email sequences
- Lead qualification
- Meeting scheduling

**Recruitment:**
- Automated candidate outreach
- Resume screening
- Interview scheduling
- Follow-up communications

**Customer Support:**
- Initial triage of support tickets
- Routing to appropriate team
- Automated responses to common questions
- Escalation handling

**Content Operations:**
- Blog post generation and formatting
- Social media content creation
- Email newsletter production
- Multi-format content conversion

**Key Insight:** Any business process that involves:
- Multiple specialized steps
- Decision-making between options
- External tool integration
- Some degree of autonomy

...can potentially be automated with multi-agent systems.

---

## Part 8: Exercises and Practice

To truly master this material, you need hands-on practice. Here are some exercises:

**Exercise 1: Identify the Patterns**
Go through the code examples and identify which agentic design patterns are being used:
- Prompt chaining?
- Routing?
- Parallelization?
- Orchestrator-worker?
- Evaluator-optimizer?

At what point did we transition from "workflows" to true "agents" in Anthropic's terminology?

**Exercise 2: Add More Tools**
Extend the sales manager example:
- Add a fact-checker tool that verifies claims in the email
- Add a personalization tool that customizes based on recipient data
- Add an A/B test tracker

**Exercise 3: Mail Merge**
Implement proper mail merge functionality:
- Read a CSV of contacts
- Generate personalized emails for each
- Track which were sent

**Exercise 4: Email Reply Handler (Hard)**
This is a significant engineering challenge:
- Set up webhooks to receive email replies
- Parse the reply
- Have the agent generate an appropriate response
- Continue the conversation thread

This requires understanding:
- Webhook setup
- Email parsing
- Conversation state management
- Async background processing

---

## Summary and Key Takeaways

Let's recap everything we've learned today:

**Asynchronous Python:**
- Foundation for all modern agent frameworks
- `async def` creates coroutines, not functions
- `await` schedules coroutines for execution
- Event loop manages concurrent execution
- `asyncio.gather()` runs multiple coroutines in parallel
- Perfect for I/O-bound operations like API calls

**OpenAI Agents SDK:**
- Lightweight, flexible, unopinionated framework
- Three core concepts: Agents, Handoffs, Guardrails
- Three steps to run: Create agent, use trace, call runner.run
- All operations are async (use await)

**Tools:**
- `@function_tool` decorator converts functions to tools
- SDK auto-generates JSON schemas from type hints and docstrings
- `.as_tool()` method wraps agents as tools
- Tools return control to the calling agent

**Handoffs:**
- Delegate control from one agent to another
- Control doesn't return to original agent
- Use `handoff_description` to advertise capability
- Different from tools: delegation vs. capability

**Vibe Coding:**
- Use LLMs to assist development
- Verify across multiple LLMs
- Work in small increments
- Validate generated code
- Request multiple approaches
- Always understand what you're using

**Practical Skills:**
- Multi-agent orchestration with asyncio.gather
- Tool creation with decorators
- External API integration (SendGrid)
- Complex workflows with tools and handoffs
- Monitoring with traces

---

## Resources and Further Learning

**Official Documentation:**
- OpenAI Agents SDK: platform.openai.com/docs/guides/agents
- Python asyncio: docs.python.org/3/library/asyncio.html

**Tools and Services:**
- SendGrid: sendgrid.com (free tier available)
- OpenAI Platform: platform.openai.com
- Trace Dashboard: platform.openai.com/traces

**Recommended Reading:**
- Anthropic's Agentic Design Patterns documentation
- Andrej Karpathy's posts on vibe coding
- Python asyncio tutorials and guides

---

## Conclusion

Today we've built a solid foundation in two critical areas:

1. **Asynchronous Python** - The underlying mechanism that makes modern agent frameworks efficient and scalable
2. **OpenAI Agents SDK** - A practical, elegant framework for building multi-agent systems

You now understand:
- How async IO works under the hood
- Why it's essential for agent development
- How to create and run agents
- How to give agents tools (both functions and other agents)
- How to coordinate agents with handoffs
- How to build complex multi-agent workflows

The key to mastery is practice. Take the examples, modify them, break them, fix them, and extend them. Build your own multi-agent systems. The patterns you've learned today are transferable to any agent framework you'll encounter.

Remember: The best way to learn is by doing. Start building!

