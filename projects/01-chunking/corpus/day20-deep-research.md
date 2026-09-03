# Day 20 Lecture Notes: Multi-Model Orchestration, Structured Outputs & Deep Research Agent

## Introduction

Welcome to Day 20! Today we're continuing our exploration of the OpenAI Agents SDK with three major topics: multi-model orchestration, structured outputs with guardrails, and building a complete Deep Research Agent. This represents the culmination of Week 2, where we transform our experimental notebook code into a production-ready application with a user interface.

## Part 1: Multi-Model Orchestration

### Why Use Multiple Models?

In real-world applications, you often want to use different AI models for different tasks. Here's why:

1. **Cost Optimization**: Some models are cheaper but still effective for simpler tasks
2. **Specialized Capabilities**: Different models excel at different things (coding, creative writing, analysis)
3. **Redundancy and Comparison**: Getting multiple perspectives on the same problem
4. **Avoiding Vendor Lock-in**: Flexibility to switch providers based on performance or pricing

The OpenAI Agents SDK makes it remarkably easy to orchestrate multiple models in a single workflow.

### OpenAI-Compatible Endpoints

Many AI providers now offer "OpenAI-compatible" endpoints. This means they implement the same API structure as OpenAI, allowing you to use the same client library with different backends.

**Providers with OpenAI-Compatible APIs:**

- **Google Gemini**: `https://generativelanguage.googleapis.com/v1beta/openai/`
- **DeepSeek**: `https://api.deepseek.com` (natively compatible)
- **Grok (xAI)**: `https://api.x.ai/v1`
- **Open Router**: Aggregator that provides access to many models through one API

### Setting Up Alternative Models

To use an alternative model, you need three pieces of information:

1. **Base URL**: The provider's API endpoint
2. **API Key**: Your authentication credential
3. **Model Name**: The specific model identifier

Here's the pattern for configuration:

```python
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel

# Create a client for the alternative provider
gemini_client = AsyncOpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GOOGLE_API_KEY")
)

# Create a model object
gemini_model = OpenAIChatCompletionsModel(
    name="gemini-2.0-flash-exp",
    client=gemini_client
)
```

### Using Model Objects in Agents

When creating an agent, you can pass either:
- **A string** (e.g., `"gpt-4o-mini"`): Assumes OpenAI's API
- **A model object**: Uses your custom endpoint

```python
from agents import Agent

# This uses OpenAI
agent1 = Agent(
    name="OpenAI Agent",
    instructions="You are a helpful assistant",
    model="gpt-4o-mini"  # String = OpenAI
)

# This uses Gemini
agent2 = Agent(
    name="Gemini Agent",
    instructions="You are a helpful assistant",
    model=gemini_model  # Object = custom endpoint
)
```

### Practical Example: Multi-Model Sales Agents

In the transcript, we saw three sales agents using different models:

```python
# Sales Agent 1: DeepSeek (cost-effective)
sales_agent_1 = Agent(
    name="Professional Sales Agent",
    instructions=professional_instructions,
    model=deepseek_model
)

# Sales Agent 2: Gemini (creative)
sales_agent_2 = Agent(
    name="Witty Sales Agent",
    instructions=witty_instructions,
    model=gemini_model
)

# Sales Agent 3: Llama 3.3 via Grok (fast)
sales_agent_3 = Agent(
    name="Concise Sales Agent",
    instructions=concise_instructions,
    model=grok_model
)
```

Each agent has the same role (writing sales emails) but uses a different underlying model. A sales manager agent can call all three in parallel and compare the results.

### Parallel Execution with Async IO

When you run multiple agents, async IO allows them to execute in parallel:

```python
import asyncio

# All three run simultaneously
results = await asyncio.gather(
    runner.run(sales_agent_1, message),
    runner.run(sales_agent_2, message),
    runner.run(sales_agent_3, message)
)
```

Total execution time ≈ the slowest agent (not the sum of all three).

### Cautionary Tale: Autonomous Agent Risks

The transcript shared an important lesson. When the instruction said "use tools multiple times if you're not satisfied with the results," one run created an infinite loop:
- 14+ tool calls
- Over 300 seconds of execution
- Kept calling the same agents repeatedly

**Lesson**: Autonomous agents need explicit constraints:
- Maximum number of iterations
- Timeout limits
- Clear success criteria
- Cost caps

### Note on Anthropic Claude

Claude does NOT offer OpenAI-compatible endpoints directly. Workarounds include:
- **Open Router**: Third-party aggregator with OpenAI-compatible API
- **MCP (Model Context Protocol)**: Anthropic's own protocol (covered in Week 6)
- Direct integration using Anthropic's SDK separately

For seamless integration with OpenAI Agents SDK, stick with OpenAI-compatible models for now.

## Part 2: Structured Outputs

### The Problem with Unstructured Text

By default, LLMs generate natural language text. This is great for human consumption but problematic for:
- Parsing and extracting specific information
- Database storage
- API integration
- Type safety and validation

**Solution**: Structured outputs force the LLM to respond in a specific schema.

### Pydantic Models

Pydantic is Python's data validation library using type hints. It's the foundation for structured outputs.

**Basic Example:**

```python
from pydantic import BaseModel

class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    """True if a personal name is found in the message"""
    
    name: str
    """The name found in the message, if any"""
```

Key features:
- Inherit from `BaseModel`
- Add type-annotated fields
- Include docstrings (the LLM sees these!)

### How Structured Outputs Work

Behind the scenes:
1. Your Pydantic model is converted to a JSON schema
2. The JSON schema is added to the LLM's prompt
3. The LLM generates JSON conforming to the schema
4. The JSON is validated and converted to a Pydantic object

You never see the JSON - you just get a properly typed Python object!

### Using Structured Outputs in Agents

```python
from agents import Agent

guardrail_agent = Agent(
    name="Name Check",
    instructions="Check if the user is including someone's personal name",
    output_type=NameCheckOutput,  # Specify the structure
    model="gpt-4o-mini"
)

# The output is automatically a NameCheckOutput object
result = await runner.run(guardrail_agent, "Send email to Alice")
print(result.is_name_in_message)  # True
print(result.name)  # "Alice"
```

### Chain of Thought with Structured Outputs

Including a "reason" field before the main output improves quality:

```python
class WebSearchItem(BaseModel):
    reason: str
    """Why this search is important for answering the query"""
    
    query: str
    """The search query to execute"""
```

Why this works:
- The LLM must articulate its reasoning first
- Subsequent fields are more coherent with the stated reason
- Similar to chain-of-thought prompting
- Exploits the sequential nature of token generation

### Complex Structured Outputs

You can nest Pydantic models:

```python
class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query"""
```

This creates a list of structured items, each with their own fields.

## Part 3: Guardrails

### What Are Guardrails?

Guardrails are validation checkpoints that protect your agent system. Unlike simple validation (checking for specific patterns), guardrails can use LLMs to make intelligent decisions about content.

**Two Types:**
- **Input Guardrails**: Validate incoming requests before processing
- **Output Guardrails**: Validate final responses before showing to users

### Why LLM-Based Guardrails?

Traditional validation is rule-based:
- Check for specific keywords
- Regex pattern matching
- Format validation

LLM-based guardrails provide semantic understanding:
- Detect PII (Personal Identifiable Information) in context
- Identify inappropriate tone or content
- Check for outdated information
- Validate business logic constraints

### Guardrail Placement Limitations

Important: Guardrails can only be placed at system boundaries:
- **Input**: Before the first agent processes anything
- **Output**: After the final agent completes

You cannot insert guardrails in the middle of workflows. This is intentional - guardrails protect the entry and exit points of your system.

### Creating a Guardrail Agent

A guardrail agent is just a regular agent with a specific purpose:

```python
class NameCheckOutput(BaseModel):
    is_name_in_message: bool
    name: str

guardrail_agent = Agent(
    name="Name Check",
    instructions="Check if the user is including someone's personal name",
    output_type=NameCheckOutput,
    model="gpt-4o-mini"
)
```

### Implementing Input Guardrails

Use the `@input_guardrail` decorator:

```python
from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def guardrail_against_name(context, agent, message):
    # Run the guardrail agent
    result = await runner.run(guardrail_agent, message)
    
    # Return validation result
    return GuardrailFunctionOutput(
        output_info={"name_found": result.name},
        tripwire_triggered=result.is_name_in_message
    )
```

**Parameters:**
- `context`: Shared data between agents (not used here)
- `agent`: The agent being guarded
- `message`: The incoming input

**Return Value:**
- `output_info`: Dictionary with metadata (appears in traces)
- `tripwire_triggered`: Boolean - if True, raises an exception

### Using Guardrails in Agents

```python
careful_agent = Agent(
    name="Careful Sales Manager",
    instructions=sales_manager_instructions,
    tools=sales_tools,
    handoffs=email_handoff,
    model="gpt-4o-mini",
    input_guardrails=[guardrail_against_name]  # Add here
)
```

### When a Guardrail Triggers

If `tripwire_triggered=True`:
1. An exception is raised immediately
2. The workflow stops
3. The trace shows which guardrail was triggered
4. No further processing occurs

This protects your system from processing invalid inputs.

### Output Guardrails

The pattern is identical, but use `@output_guardrail` and receive `output` instead of `message`:

```python
@output_guardrail
async def guardrail_against_bad_output(context, agent, output):
    # Validate the output
    # Return GuardrailFunctionOutput
    pass
```

Use cases:
- Check for hallucinations
- Validate tone and professionalism
- Ensure completeness
- Verify no sensitive data is leaked

## Part 4: Deep Research Agent

### Overview

The Deep Research Agent is a classic agentic use case that:
1. Takes a research query from the user
2. Plans multiple web searches to investigate the topic
3. Executes searches in parallel
4. Synthesizes findings into a comprehensive report
5. Emails the report in HTML format

This is what OpenAI, Perplexity, and other companies offer as "Deep Research" features.

### Why Build Your Own?

- Customize for your specific domain or business
- Control costs and search depth
- Add proprietary data sources
- Integrate with your existing tools
- Learn the architecture for other agentic applications

The architecture is universal and can be adapted to any domain.

### OpenAI Hosted Tools

OpenAI provides three "hosted" tools (they run on OpenAI's infrastructure):

1. **Web Search Tool**: Search the internet
2. **File Search Tool**: Search across vector stores
3. **Computer Tool**: Take screenshots, click around

We'll use the Web Search Tool for our Deep Research Agent.

### Web Search Tool Costs

**Important**: As of early 2025, the Web Search Tool costs:
- GPT-4o-mini with low search context: **$0.025 per search**
- 10 searches = $0.25
- 20 searches = $0.50

**Search context sizes:**
- **Low**: Cheapest, sufficient for most needs
- **Medium**: Default, slightly more expensive
- **High**: Most expensive, most comprehensive

A comprehensive research session can cost $1-3. Monitor your usage!

Check current pricing: https://openai.com/api/pricing/

### Architecture Overview

The Deep Research Agent has five key components:

1. **Planner Agent**: Converts query into list of searches
2. **Search Agent**: Executes web searches using hosted tool
3. **Writer Agent**: Synthesizes search results into report
4. **Email Agent**: Formats and sends the final report
5. **Research Manager**: Orchestrates the entire workflow

Each component has a single, clear responsibility (separation of concerns).

### Component 1: Planner Agent

**Purpose**: Convert a research query into a structured list of search queries.

**Structured Output:**

```python
class WebSearchItem(BaseModel):
    reason: str
    """Why this search is important for answering the query"""
    query: str
    """The search query to execute"""

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform"""
```

**Agent Definition:**

```python
NUM_SEARCHES = 3  # Configurable (3 for cost control, 20 for depth)

planner_instructions = f"""
You are a helpful research assistant.
Given a query, come up with a set of web searches to perform to best answer that query.
Output {NUM_SEARCHES} terms to query for.
"""

planner_agent = Agent(
    name="Planner Agent",
    instructions=planner_instructions,
    output_type=WebSearchPlan,
    model="gpt-4o-mini"
)
```

**Usage:**

```python
plan = await runner.run(planner_agent, "Latest AI agent frameworks in 2025")
# plan.searches is a list of WebSearchItem objects
```

### Component 2: Search Agent

**Purpose**: Execute web searches using OpenAI's hosted Web Search Tool.

**Instructions** (from OpenAI's documentation):

```python
search_instructions = """
You're a research assistant.
Given a search term, you search the web for that term.
You produce a concise summary (2-3 paragraphs) that captures the main points.
Write succinctly. No need for good grammar as it will be consumed by someone else synthesizing a report.
"""
```

**Agent Definition:**

```python
from agents import WebSearchTool

search_agent = Agent(
    name="Search Agent",
    instructions=search_instructions,
    tools=[WebSearchTool(search_context_size="low")],  # Low for cost
    model="gpt-4o-mini",
    model_settings={"tool_choice": "required"}  # Must use the tool
)
```

**Key Configuration:**
- `search_context_size="low"`: Cheapest option
- `tool_choice="required"`: Agent must use the search tool
- Model: GPT-4o-mini (cheapest model)

### Component 3: Parallel Search Execution

**Purpose**: Run all searches in parallel for efficiency.

```python
async def perform_searches(search_plan: WebSearchPlan):
    # Create async tasks for each search
    tasks = [
        asyncio.create_task(search(item))
        for item in search_plan.searches
    ]
    
    # Run all in parallel
    results = await asyncio.gather(*tasks)
    return results

async def search(item: WebSearchItem):
    # Provide context with the reason
    message = f"Search for: {item.query}\nReason: {item.reason}"
    result = await runner.run(search_agent, message)
    return result.final_output
```

**Result**: All searches complete in approximately the time of the slowest one.

### Component 4: Writer Agent

**Purpose**: Synthesize all search results into a comprehensive report.

**Structured Output:**

```python
class ReportData(BaseModel):
    short_summary: str
    """A short 2-3 sentence summary"""
    
    markdown_report: str
    """The full final report in markdown format"""
    
    follow_up_suggestions: list[str]
    """Suggested topics to research further"""
```

**Agent Definition:**

```python
writer_instructions = """
You are a senior researcher tasked with writing a cohesive report for a query.
You'll be provided with the original query and initial research done by a research assistant.
Come up with an outline for the report that describes the structure and flow.
Then generate the report and return that as your final output.
Should be in markdown, lengthy and detailed (5-10 pages, at least 1000 words).
"""

writer_agent = Agent(
    name="Writer Agent",
    instructions=writer_instructions,
    output_type=ReportData,
    model="gpt-4o-mini"
)
```

**Usage:**

```python
report = await runner.run(
    writer_agent,
    f"Query: {query}\n\nResearch Results:\n{search_results}"
)
# report.markdown_report contains the full report
```

### Component 5: Email Agent

**Purpose**: Convert the markdown report to HTML and send via email.

**Tool Definition:**

```python
from agents import function_tool
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

@function_tool
def send_email(subject: str, body: str):
    """Send an email with the given subject and HTML body"""
    message = Mail(
        from_email="your-verified@email.com",
        to_emails="recipient@email.com",
        subject=subject,
        html_content=body
    )
    sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
    response = sg.send(message)
    return f"Email sent! Status: {response.status_code}"
```

**Agent Definition:**

```python
email_instructions = """
You're able to send a nicely formatted HTML email based on a detailed report.
You'll be provided with a detailed report. Use your tool to send one email providing the report.
Convert it into clean, well-presented HTML with a subject line.
"""

email_agent = Agent(
    name="Email Agent",
    instructions=email_instructions,
    tools=[send_email],
    model="gpt-4o-mini"
)
```

### Component 6: Research Manager

**Purpose**: Orchestrate all agents in the correct sequence.

```python
async def run_research(query: str):
    # Step 1: Plan searches
    search_plan = await runner.run(planner_agent, query)
    
    # Step 2: Perform searches in parallel
    search_results = await perform_searches(search_plan)
    
    # Step 3: Write report
    report = await runner.run(
        writer_agent,
        f"Query: {query}\n\nResults: {search_results}"
    )
    
    # Step 4: Send email
    await runner.run(email_agent, report.markdown_report)
    
    return report
```

### Live Progress Updates with Generators

To show progress in real-time, use Python generators with `yield`:

```python
async def run_research(query: str):
    yield "Planning searches..."
    search_plan = await runner.run(planner_agent, query)
    
    yield f"Performing {len(search_plan.searches)} searches..."
    search_results = await perform_searches(search_plan)
    
    yield "Writing report..."
    report = await runner.run(writer_agent, ...)
    
    yield "Sending email..."
    await runner.run(email_agent, report.markdown_report)
    
    yield report.markdown_report  # Final result
```

Gradio will display each yielded value in real-time.

## Part 5: From Notebook to Production

### Why Start with Notebooks?

Notebooks are great for:
- Rapid experimentation
- Iterating on prompts
- Testing agent architectures
- Visualizing results

But production code needs:
- Type hints
- Exception handling
- Modular structure
- Reusability

### Transition Strategy

1. **Experiment in notebooks**: Rapid iteration
2. **Extract to modules**: Create clean, typed code
3. **Add UI**: Wrap with Gradio for user-friendly interface

### Module Structure

```
deep_research/
├── planner_agent.py      # Planner agent definition
├── search_agent.py       # Search agent definition
├── writer_agent.py       # Writer agent definition
├── email_agent.py        # Email agent definition
├── research_manager.py   # Orchestration logic
└── deepresearch.py       # Gradio UI
```

Each agent gets its own module for clarity and reusability.

### Example Module: Planner Agent

```python
# planner_agent.py
from pydantic import BaseModel
from agents import Agent

NUM_SEARCHES = 3

class WebSearchItem(BaseModel):
    reason: str
    """Why this search is important"""
    query: str
    """The search query to execute"""

class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """List of web searches to perform"""

planner_instructions = f"""
You are a helpful research assistant.
Given a query, come up with {NUM_SEARCHES} web searches to perform.
"""

planner_agent = Agent(
    name="Planner Agent",
    instructions=planner_instructions,
    output_type=WebSearchPlan,
    model="gpt-4o-mini"
)
```

Clean, focused, and reusable!

### Building the Gradio UI

Gradio makes it easy to create web interfaces without frontend knowledge.

**Basic Structure:**

```python
import gradio as gr

def create_ui():
    with gr.Blocks() as ui:
        gr.Markdown("# Deep Research")
        
        query_textbox = gr.Textbox(
            label="What topic would you like to research?",
            placeholder="Enter your research query..."
        )
        
        run_button = gr.Button("Run", variant="primary")
        
        report = gr.Markdown(label="Report")
        
        # Connect events
        run_button.click(
            fn=run_research,
            inputs=query_textbox,
            outputs=report
        )
    
    return ui

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(inbrowser=True)
```

**Key Components:**
- `gr.Blocks()`: Custom layout container
- `gr.Textbox()`: Input field
- `gr.Button()`: Action trigger
- `gr.Markdown()`: Display area
- `.click()`: Event handler

### Generator Callbacks for Live Updates

```python
async def run_research(query: str):
    """Generator that yields progress updates"""
    yield "Starting research..."
    
    # ... perform research ...
    
    yield "Complete!"
    yield final_report
```

Gradio automatically displays each yielded value in sequence.

### Running the Application

```bash
# With UV (recommended)
uv run deepresearch.py

# Or with Python directly
python deepresearch.py
```

The UI opens at `http://localhost:7860`

### Deployment to Hugging Face Spaces

```bash
gradio deploy
```

This deploys your app to Hugging Face Spaces for free hosting. You get a public URL to share!

## Part 6: Tracing and Debugging

### OpenAI Trace Viewer

Every run generates a trace ID. View at:
```
https://platform.openai.com/traces/{trace_id}
```

**What the trace shows:**
- Each agent that ran
- Tools called and their results
- Timing information
- Parallel vs. sequential execution
- Token usage and costs
- Input/output for each step

**Generating trace links:**

```python
import uuid

trace_id = str(uuid.uuid4())
trace_url = f"https://platform.openai.com/traces/{trace_id}"
print(f"View trace: {trace_url}")

# Use with trace context
with trace(trace_id):
    result = await runner.run(agent, message)
```

### Debugging Parallel Execution

In the trace, you can see:
- Which agents ran in parallel
- How long each took
- Which was the bottleneck

This helps optimize your workflow.

## Part 7: Making It Better

### Enhancement 1: Clarifying Questions

Like OpenAI's Deep Research, ask the user for clarification first:

```python
class ClarifyingQuestions(BaseModel):
    questions: list[str]
    """3 clarifying questions to ask the user"""

clarifier_agent = Agent(
    name="Clarifier",
    instructions="Given a query, ask 3 clarifying questions",
    output_type=ClarifyingQuestions,
    model="gpt-4o-mini"
)
```

### Enhancement 2: Autonomous Manager

Instead of a fixed workflow, let an agent decide what to do next:

```python
manager_agent = Agent(
    name="Research Manager",
    instructions="Coordinate research. Decide if more searches are needed.",
    tools=[plan_searches_tool, perform_search_tool, write_report_tool],
    model="gpt-4o-mini"
)
```

The manager has autonomy to:
- Decide how many searches to run
- Determine if results are sufficient
- Request refined searches based on findings

### Enhancement 3: Evaluator-Optimizer Pattern

Add an agent to review and improve results:

```python
evaluator_agent = Agent(
    name="Evaluator",
    instructions="Review the report. Identify gaps and suggest improvements.",
    output_type=EvaluationResult,
    model="gpt-4o-mini"
)
```

### Enhancement 4: Query Refinement

Update searches based on initial findings:

```python
# After first round of searches
refined_plan = await runner.run(
    planner_agent,
    f"Original query: {query}\nFindings so far: {initial_results}\nWhat else should we search?"
)
```

### Enhancement 5: Custom Data Sources

Beyond web search, add:
- Internal document search
- Database queries
- API calls to proprietary systems
- Expert interviews (human-in-the-loop)

## Conclusion

Today we covered:
- **Multi-model orchestration**: Using different AI models in one workflow
- **Structured outputs**: Forcing LLMs to return specific data structures
- **Guardrails**: Protecting inputs and outputs with validation agents
- **Deep Research Agent**: A complete, production-ready agentic application
- **Production practices**: Converting notebooks to modules with Gradio UI

You now have the skills to build sophisticated, multi-agent applications that are ready for real-world use!

### Week 2 Complete!

This week we mastered:
- Day 17: Agent fundamentals and design patterns
- Day 18: Multi-model orchestration and career agent
- Day 19: Async IO and OpenAI Swarm framework
- Day 20: Structured outputs, guardrails, and Deep Research

**Next Week**: Crew AI - A different agent framework with role-based design!

### Resources

- OpenAI Agents SDK: https://platform.openai.com/docs/agents
- OpenAI Pricing: https://openai.com/api/pricing/
- Gradio Documentation: https://gradio.app/
- Pydantic Documentation: https://docs.pydantic.dev/
- Google Gemini: https://ai.google.dev/
- DeepSeek: https://www.deepseek.com/
- Grok (xAI): https://x.ai/

Keep experimenting, keep building, and see you in Week 3!

