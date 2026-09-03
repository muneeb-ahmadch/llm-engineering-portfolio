# Day 18: Tools, Resources & Building Your Career Agent - Lecture Notes

## Introduction

Welcome to Day 18! This is where theory meets practice in an exciting way. In Day 17, we learned about the theoretical foundations of AI agents - what they are, the different patterns, and how they work conceptually. Today, we're rolling up our sleeves and actually building a real, deployable AI agent from scratch.

This session is special because we're not using any frameworks yet. We're working directly with LLM APIs to understand exactly what's happening under the hood. Think of it like learning to drive a manual transmission car before moving to an automatic - you'll understand the mechanics much better, which will make you a better developer when we start using frameworks next week.

## Part 1: Understanding the AI Agent Framework Landscape

### Why Talk About Frameworks If We're Not Using Them?

Good question! Before we dive into building without frameworks, it's important to understand the landscape of what's available. This helps you make informed decisions about when to use what approach.

Think of it like cooking. You can make bread from scratch (no framework), use a bread machine (lightweight framework), or buy pre-made dough (heavyweight framework). Each has its place depending on your needs, skills, and time constraints.

### The Framework Hierarchy

Let's explore the different levels of complexity in the AI agent framework world:

**Level 1: No Framework (What We're Doing This Week)**

At the bottom of the complexity hierarchy - or perhaps we should say the foundation - is using no framework at all. You connect directly to LLMs using their APIs, just like we did in Day 17's lab.

**Why This Approach?**
- **Complete Control:** You see exactly what's happening at every step
- **No Abstractions:** No "magic" hiding important details
- **Learning:** Best way to understand how agents actually work
- **Flexibility:** Can do anything without framework limitations

**Anthropic's Perspective:**
In their blog post "Building Effective Agents," Anthropic makes a compelling case for this approach. They argue that the APIs are relatively simple and straightforward, and the benefit is you get to see exactly what's going on under the hood. You control the prompts in detail, which gives you maximum flexibility.

**When to Use:**
- Learning and understanding fundamentals
- Simple applications
- When you need maximum control
- When framework overhead isn't justified

**Level 2: Lightweight Frameworks**

These frameworks provide helpful abstractions while still keeping things relatively simple.

**OpenAI Agents SDK (Week 2):**
This is one of the instructor's favorites. It's super lightweight, simple, clean, and flexible. It was released very recently (so new that the API changed during course development!), but it's really great.

Think of it as a thin wrapper around the core concepts. You still feel like you're working with LLMs, but it handles a lot of the boilerplate code for you.

**CrewAI (Week 3):**
Another favorite, CrewAI has been around longer and is very easy to use. It has a "low-code" angle - you can do a lot through configuration using YAML files rather than writing lots of Python code.

Imagine assembling a crew of specialists to solve a problem. Each crew member has a role, and you configure how they work together. It's intuitive and powerful.

**Level 3: Heavyweight Frameworks**

These are the big guns - powerful but complex.

**LangGraph (Week 4):**
From the creators of LangChain, LangGraph is sophisticated and quite complex. It has a steep learning curve, particularly compared to the lightweight frameworks.

The core idea is that you're building a computational graph of your agents and their tools. It's very powerful and means you can build quite sophisticated things, but there's a cost in terms of learning curve. You're signing up for an ecosystem with lots of terminology, concepts, and abstractions.

When you use LangGraph, your project becomes a "LangGraph project" rather than just an "AI project." The framework takes over in a big way.

**Autogen (Week 5):**
Microsoft's Autogen is actually a couple of different things (we'll explore all of them). One aspect is like an environment where agents can collaborate remotely. It's powerful but, like LangGraph, comes with significant complexity.

**The Key Difference:**
With OpenAI Agents SDK and CrewAI, you still feel like you're just interacting with LLMs. With LangGraph and Autogen, you're very much part of their ecosystem. It's a different experience.

**Special Mention: MCP (Model Context Protocol) - Week 6**

MCP isn't really a framework - it's a protocol. Created by Anthropic (who, remember, advocates for no frameworks), it's an open-source way for different models to connect and collaborate, sharing each other's capabilities using a common protocol.

Think of it like USB-C for AI models. Instead of needing custom adapters (code) for each connection, you use a standard protocol. As long as you conform to this protocol, you can stitch together models and their providers in a very elegant, simple way.

### Making Your Choice

**Consider These Factors:**

**1. Use Case:**
Different frameworks fit different business objectives better. A simple chatbot might not need LangGraph's power, while a complex multi-agent system might benefit from it.

**2. Personal Preference:**
The instructor admits a bias toward the simpler, lightweight options. They "stay out of your way" and are simple and flexible. But there's also appreciation for the power of the heavyweight frameworks.

**3. Team Skills:**
Consider what your team knows and can learn. A team familiar with complex frameworks might prefer them, while a team new to AI might benefit from starting simple.

**4. Project Requirements:**
How complex is your use case? How much control do you need? How important is development speed vs. flexibility?

**The Instructor's Recommendation:**
Start at the bottom and work your way up. Understand the fundamentals first, then add complexity as needed. This course is designed to give you the full spectrum, equipping you to choose wisely.

## Part 2: Resources - Enhancing LLM Capabilities

### What Are Resources?

Resources is really just a fancy way of saying you can improve the effectiveness of an LLM by providing it with more context and information to improve its expertise.

Let's break this down with an example.

### The Basic Concept

Imagine you're building a customer support chatbot for an airline. If you just prompt an LLM with "Answer this customer's question," it will do its best, but it won't know your specific ticket prices, policies, or procedures.

**Without Resources:**
```
User: "How much is a flight to Paris?"
LLM: "I don't have access to current flight prices..."
```

**With Resources:**
```
System Prompt: You are a customer support agent.
Resources: [All ticket prices, routes, policies]

User: "How much is a flight to Paris?"
LLM: "A flight to Paris costs $450 for economy or $1,200 for business class..."
```

See the difference? The LLM now has the information it needs to answer accurately and helpfully.

### How It Works

The mechanism is beautifully simple:
1. Identify what information would be helpful
2. Grab that relevant data
3. Shove it into the prompt you send to the LLM
4. The LLM can now reference this information in its responses

That's it! You're literally just putting extra text in the prompt.

### Beyond Simple Insertion

Now, when I say "it's nothing more fancy than that," there actually IS a bit more fancy than that. You can use clever techniques to make this more sophisticated:

**Selective Context:**
Instead of including ALL ticket prices (which might be thousands of entries), you can:
- Analyze the question
- Determine which prices are relevant
- Only include those specific prices

**Using LLMs to Help:**
You can even use other LLMs to help figure out the best, most relevant context. For example:
1. User asks a question
2. One LLM analyzes the question and identifies key topics
3. You retrieve documents related to those topics
4. Another LLM summarizes those documents
5. You include the summary as resources

**RAG (Retrieval Augmented Generation):**
This is a whole field of study! RAG is about retrieving relevant context intelligently. It's covered extensively in the instructor's other course (LLM Engineering), but the basic idea is:
- Store information in a vector database
- When a question comes in, find the most relevant information
- Include only that relevant information as resources

### Real-World Examples

**Customer Support:**
- Product documentation
- Company policies
- FAQ databases
- Previous ticket resolutions
- Troubleshooting guides

**Legal Assistant:**
- Relevant case law
- Statutes and regulations
- Previous case outcomes
- Client-specific information
- Contract templates

**Medical Assistant:**
- Patient history
- Medical guidelines
- Drug interaction databases
- Treatment protocols
- Research papers

**Our Career Chatbot:**
- LinkedIn profile (exported as PDF)
- Resume/CV
- Personal summary with fun facts
- Project descriptions
- Skills and achievements
- Work history

### The Power of Resources

Resources transform a generic LLM into a domain expert. A base GPT-4 knows general information about many topics, but it doesn't know:
- Your company's specific policies
- Your personal career history
- Your product's technical specifications
- Your customer's account details

By providing these as resources, you create a specialized expert that can answer questions accurately and helpfully in your specific domain.

### Practical Considerations

**Token Limits:**
LLMs have context window limits (how much text they can process at once). You need to be strategic about what you include.

**Cost:**
More tokens = more cost. Including huge documents in every prompt can get expensive.

**Relevance:**
Including irrelevant information can actually confuse the LLM or cause it to focus on the wrong things.

**Freshness:**
Resources need to be kept up-to-date. Outdated information leads to incorrect answers.

## Part 3: Tools - Giving LLMs the Power to Act

### The Core Concept

Tools are where things get really exciting. Tools give an LLM the power to DO something, to use a tool, and crucially, to use that tool at its discretion.

This is one of the key tricks to giving an LLM autonomy. Instead of just generating text, the LLM can take actions in the real world.

### What Are Tools?

A tool is any function or capability you make available to an LLM. Examples:
- Query a database with SQL
- Send a message or notification
- Search the web
- Read or write files
- Call an external API
- Control IoT devices (like lights!)
- Make calculations
- Book appointments
- Update a calendar
- Send emails

The list is endless. If you can write a Python function to do it, you can make it a tool.

### The "Magic" of Tool Calling

When you first hear about tools and function calling, it might sound kind of crazy. You're interacting with OpenAI on the cloud, and you're saying "hey, you have a tool you can use - you can query my database." And it's going to be able to connect back to your computer and query the database as part of giving its response?

That sounds magical, maybe even creepy. How exactly is it going to do that?

### The Reality: It's a Conjuring Trick

The truth is actually rather more mundane. People who know about tools know this already, but it has a sense of being very magical. The reality is a bit of a conjuring trick.

**What It Seems Like (Theory):**
```
Your Code → Prompt → LLM on Cloud → Executes Tool → Response
```

The LLM magically reaches back to your computer and runs your function!

**What Actually Happens (Practice):**
```
Your Code → Prompt + Tool Descriptions → LLM → Response: "I want to use tool X"
Your Code → Checks response → Runs tool X → Sends result back to LLM
Your Code → LLM → Final response
```

Let me break this down step by step:

**Step 1: You Describe Available Tools**
In your prompt to the LLM, you list out everything it's able to ask for. You say something like:

```
"I would like you to tell me if you want me to take care of a few actions 
that I can do on your behalf. One of them is called 'turn_on_lights'. 
If you reply 'turn_on_lights', then I will do that and I will get back 
to you with confirmation."
```

**Step 2: LLM Responds**
The LLM responds in JSON format (the tool calling code packages this away from you, but that's what's really happening). In that JSON response, it says what it wants to do.

**Step 3: You Check the Response**
You have to write an if statement in your code (you can be fancier than an if statement, but it's like an if statement):

```python
if llm_wants_to_do_X:
    do_X()
    call_llm_again_with_results()
```

**Step 4: You Run the Tool**
Your code executes the actual function.

**Step 5: You Call the LLM Again**
You send a second request to the LLM, this time including the results of running the tool.

**Step 6: LLM Generates Final Response**
Now the LLM can incorporate the tool results into its answer.

### A Concrete Example

Let me show you a real conversation to make this concrete:

**Prompt to GPT-4:**
```
You're a support agent for an airline.
You answer users' questions.
You have the ability to query ticket prices.
Just respond: "Use tool to fetch ticket price for [city]"

Here's the user question:
"I'd like to go to Paris. How much is a flight?"
```

**ChatGPT's Reply:**
```
Use tool to fetch ticket price for Paris
```

That's ALL there is to it! 

You write code that:
1. Prompts a bit like this
2. Gets back "Use tool to fetch ticket price for Paris"
3. Parses that response
4. Calls your `fetch_ticket_price("Paris")` function
5. Sends the question a second time with the answer included
6. Gets back the final response to the user

You've just seen tool use in action!

### The "Autonomy" Part

We can interpret this by saying we're giving ChatGPT the autonomy to decide that it wants to make use of this tool should it wish to. That sounds very mystical and powerful.

But at the end of the day, it's JSON and if statements. That's how it works.

### Why This Matters

Understanding this removes the mystery. You're not giving the LLM actual access to your computer. You're not creating security risks. You're simply:
1. Describing what you CAN do
2. Asking the LLM if it WANTS you to do it
3. Doing it if it asks
4. Telling it the results

It's a conversation, not magic.

### The Power Despite Simplicity

Even though the mechanism is simple, the results are powerful. The LLM can:
- Decide WHEN to use a tool (not every question needs it)
- Decide WHICH tool to use (if you offer multiple)
- Decide WHAT parameters to pass
- Chain multiple tool calls together
- Adapt based on tool results

This creates genuinely intelligent, adaptive behavior from a relatively simple mechanism.

## Part 4: Building the Career Conversation Agent

Now let's put everything together and build a real, deployable application!

### Project Overview

We're building a career chatbot that:
- Answers questions about your professional background
- Uses resources (your LinkedIn profile, resume, personal summary)
- Uses tools (to record user interest, track unknown questions)
- Can be deployed to your website
- Serves as an interactive alternative to a traditional resume

### Why This Project Matters

**Professionally:**
This is genuinely useful! You can deploy this on your personal website. Instead of a static resume, people can have a conversation with an AI that knows about your career.

**Educationally:**
This project teaches you:
- How to use resources effectively
- How to implement tool calling from scratch
- How to structure an agent application
- How to deploy to production

**Commercially:**
The patterns you learn here apply to any business chatbot or agent application. Customer support, sales assistants, internal tools - they all use these same techniques.

### Step 1: Setting Up the Project Structure

**Directory Organization:**
```
foundations/
├── me/
│   ├── linkedin.pdf      # Your LinkedIn profile
│   └── summary.txt       # Personal summary
├── app.py               # Main application
└── .env                 # Environment variables
```

**The "me" Folder:**
This is where you put information about yourself. The instructor has their LinkedIn PDF and a summary text file. You should replace these with your own!

**Getting Your LinkedIn PDF:**
1. Go to LinkedIn
2. Click on your profile
3. Find the "More" menu (three dots)
4. Select "Save to PDF"
5. Download and save as `linkedin.pdf`

**Creating Your Summary:**
Write a brief summary about yourself. Include:
- Current role and company
- Key achievements
- Areas of expertise
- A fun fact or personal touch

Example:
```
I'm currently the CTO and co-founder of Nebula, an AI startup focused on 
transforming talent sourcing. Previously, I spent 15 years at JP Morgan 
running engineering teams. I hold a patent for machine learning applications 
in financial services. Fun fact: I'm terrible at anything requiring hand-eye 
coordination but excellent at AI engineering!
```

### Step 2: Loading Resources

**Reading the LinkedIn PDF:**

```python
import pypdf2

# Create a PDF reader
reader = pypdf2.PdfReader("me/linkedin.pdf")

# Extract text from all pages
linkedin_text = ""
for page in reader.pages:
    linkedin_text += page.extract_text()
```

**PyPDF2:**
This is a popular Python library for working with PDF files. It can:
- Read PDF content
- Extract text
- Get metadata
- Merge PDFs
- Split PDFs

For our purposes, we just need to extract text.

**Reading the Summary:**

```python
with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()
```

**Note for Windows Users:**
The `encoding="utf-8"` parameter is sometimes needed on Windows to handle special characters properly. It's covered in the course guides if you run into issues.

### Step 3: Creating the System Prompt

Now we combine everything into a system prompt:

```python
name = "Your Name"  # Change this to your name!

system_prompt = f"""
You are acting as {name}.
You are answering questions on that person's website, particularly 
questions related to their career, background, skills, and experience.

Your responsibility is to represent {name} for interactions on the 
website as faithfully as possible.

You are given a summary of their background and their LinkedIn profile.
Use this information to answer questions accurately.

Be professional and engaging. If you don't know the answer to a question, 
say so clearly.

## Summary
{summary}

## LinkedIn Profile
{linkedin_text}

With this context, please chat with the user, always staying in character 
as {name}.
"""
```

**What's Happening Here:**
- We're using an f-string to insert variables
- The `{name}` gets replaced with your actual name
- The `{summary}` and `{linkedin_text}` get replaced with your content
- We're using markdown headers (##) to organize the information
- This entire string becomes the system prompt

**Why This Works:**
The LLM now has all the context it needs to answer questions about your career as if it were you. It's like you've given it your resume and told it to answer questions on your behalf.

### Step 4: System vs User Prompts

This is a good time to understand the difference between system and user prompts.

**System Prompt:**
- Sets the overall context and instructions
- Defines the agent's role and behavior
- Includes resources and background information
- Sets the tone (professional, friendly, formal, etc.)
- Remains constant throughout the conversation

**User Prompt:**
- The actual question from the user
- Changes with each message
- Drives the conversation forward
- What the user wants to know

**Why Separate Them?**
- **Clarity:** Easier to understand and maintain
- **Efficiency:** System prompt sent once, user prompts change
- **Performance:** Models perform better with clear role definition
- **Standard Practice:** This is how modern LLM applications are structured

**Example Conversation:**
```
System: "You are a helpful career assistant for John Doe..."
User: "What's your educational background?"
Assistant: "I have a BS in Computer Science from MIT..."
User: "What's your greatest achievement?"
Assistant: "My greatest achievement was co-founding..."
```

The system prompt stays the same; only the user prompts change.

### Step 5: Creating a Simple Chat Interface

Before we add tools, let's create a basic chat interface using Gradio.

**What is Gradio?**
Gradio is a Python library that makes it incredibly easy to build user interfaces for machine learning applications. Even if you're terrible at frontend development (like the instructor admits to being!), you can create beautiful, functional interfaces.

**Basic Gradio Chat:**

```python
import gradio as gr

def chat(message, history):
    # Build messages list
    messages = [
        {"role": "system", "content": system_prompt},
        *history,  # Spread operator to include all history
        {"role": "user", "content": message}
    ]
    
    # Call OpenAI
    response = openai.chat.completions.create(
        model="gpt-4-mini",
        messages=messages
    )
    
    # Return the response
    return response.choices[0].message.content

# Create and launch interface
gr.ChatInterface(fn=chat).launch()
```

**What's Happening:**
1. `chat` function takes the current message and conversation history
2. We build the full messages list (system + history + current)
3. We call OpenAI's API
4. We return the response
5. Gradio handles all the UI automatically!

**The `*history` Syntax:**
This is Python's "spread" or "unpacking" operator. If `history` is a list of messages, `*history` unpacks them all into the messages list. It's equivalent to:
```python
messages = [system_message]
for msg in history:
    messages.append(msg)
messages.append(user_message)
```

But much more concise!

**Running It:**
When you run this, Gradio creates a web interface with:
- A chat input box
- A conversation history display
- Automatic message handling
- Clean, professional appearance

You can test it immediately by asking questions about your career!

### Step 6: Introducing Tools

Now let's add the ability for our agent to take actions.

**Our Two Tools:**

**1. Record User Details:**
When someone wants to get in touch, record their information.

**2. Record Unknown Question:**
When the agent doesn't know an answer, record the question so you can improve later.

**Why These Tools?**
- **Practical:** You actually want to know when people are interested!
- **Simple:** Easy to understand and implement
- **Demonstrative:** Show the full power of tool calling

### Step 7: Pushover for Notifications

Before we implement the tools, we need a way to notify ourselves. Enter Pushover!

**What is Pushover?**
Pushover is a simple service that lets you send push notifications to your phone. It's much easier than SMS (which has lots of regulations now) and it's free for the first month.

**Setting Up Pushover:**
1. Go to https://pushover.net
2. Create an account
3. Install the Pushover app on your phone
4. Get your User Key and create an API Token
5. Add these to your `.env` file:
   ```
   PUSHOVER_USER=your_user_key
   PUSHOVER_TOKEN=your_api_token
   ```

**The Push Function:**

```python
import requests
import os

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"

def push(message):
    """Send a push notification to your phone"""
    requests.post(PUSHOVER_URL, data={
        "user": os.environ["PUSHOVER_USER"],
        "token": os.environ["PUSHOVER_TOKEN"],
        "message": message
    })
```

**Testing It:**

```python
push("Hey!")  # Your phone should buzz!
```

It's that simple! Now you can get notifications from your code.

### Step 8: Implementing Tool Functions

Now let's create the actual Python functions that our tools will call:

**Record User Details:**

```python
def record_user_details(email, name="", notes=""):
    """
    Record when a user wants to get in touch.
    
    Args:
        email: User's email address (required)
        name: User's name (optional)
        notes: Additional notes (optional)
    
    Returns:
        Confirmation message
    """
    message = f"Recording interest from {name} ({email}): {notes}"
    push(message)
    return "Recorded successfully"
```

**Record Unknown Question:**

```python
def record_unknown_question(question):
    """
    Record a question the agent couldn't answer.
    
    Args:
        question: The question that couldn't be answered
    
    Returns:
        Confirmation message
    """
    message = f"Recording question I couldn't answer: {question}"
    push(message)
    return "Question recorded"
```

**What These Do:**
- Format a notification message
- Send it to your phone via Pushover
- Return a confirmation (the LLM will see this)

These are simple functions, but they demonstrate the concept. In a real application, you might:
- Save to a database
- Send an email
- Create a CRM entry
- Trigger a workflow

### Step 9: Describing Tools in JSON

Now comes the verbose part. We need to describe these tools in JSON format so the LLM understands what they do.

**Tool Description for record_user_details:**

```python
record_user_details_json = {
    "type": "function",
    "function": {
        "name": "record_user_details",
        "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The user's email address"
                },
                "name": {
                    "type": "string",
                    "description": "The user's name if provided"
                },
                "notes": {
                    "type": "string",
                    "description": "Any additional notes or context"
                }
            },
            "required": ["email"],
            "additionalProperties": False
        }
    }
}
```

**Understanding This JSON:**

**`type: "function"`:**
We're describing a function the LLM can call.

**`name`:**
Must match the actual Python function name exactly.

**`description`:**
This is crucial! The LLM uses this to decide when to use the tool. Be clear and specific.

**`parameters`:**
Describes what information the function needs.

**`properties`:**
Each parameter with its type and description.

**`required`:**
Which parameters must be provided (email is required, name and notes are optional).

**Why So Verbose?**
This JSON is what gets sent to the LLM. It needs to be detailed enough for the LLM to understand:
- What the tool does
- When to use it
- What information it needs
- What's required vs optional

**The Good News:**
Frameworks automate this! Next week with OpenAI Agents SDK, you won't have to write this JSON manually. But understanding it now is valuable.

**Tool Description for record_unknown_question:**

```python
record_unknown_question_json = {
    "type": "function",
    "function": {
        "name": "record_unknown_question",
        "description": "Use this tool when you don't know the answer to a question. Record the question so it can be answered later.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question that couldn't be answered"
                }
            },
            "required": ["question"],
            "additionalProperties": False
        }
    }
}
```

**Combining Into Tools List:**

```python
tools = [
    record_user_details_json,
    record_unknown_question_json
]
```

This list of tool descriptions is what we'll pass to the OpenAI API.

### Step 10: Handling Tool Calls

This is the most important part - the "glorified if statement" that makes everything work.

**The handle_tool_calls Function:**

```python
import json

def handle_tool_calls(tool_calls):
    """
    Execute the tools the LLM wants to use.
    
    Args:
        tool_calls: List of tool calls from the LLM response
    
    Returns:
        List of messages with tool results
    """
    messages = []
    
    for tool_call in tool_calls:
        # Extract tool information
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # The if statement approach
        if tool_name == "record_user_details":
            result = record_user_details(**arguments)
        elif tool_name == "record_unknown_question":
            result = record_unknown_question(**arguments)
        else:
            result = "Unknown tool"
        
        # Add result to messages
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
    
    return messages
```

**Understanding This Function:**

**Loop Through Tool Calls:**
The LLM might request multiple tools (though usually just one at a time).

**Extract Information:**
- `tool_name`: Which function to call
- `arguments`: What parameters to pass (comes as JSON string)

**The If Statement:**
This is the "glorified if statement" we talked about! We check which tool was requested and call the corresponding function.

**The `**arguments` Syntax:**
This unpacks the dictionary into keyword arguments. If `arguments = {"email": "test@example.com", "name": "John"}`, then `**arguments` becomes `email="test@example.com", name="John"`.

**Return Format:**
We return messages in a specific format:
- `role: "tool"`: Indicates this is a tool result
- `tool_call_id`: Links this result to the original request
- `content`: The result from running the tool

**The Pythonic Alternative:**

Instead of if statements, we can use Python's `globals()` dictionary:

```python
def handle_tool_calls(tool_calls):
    messages = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Dynamic function lookup!
        tool_function = globals()[tool_name]
        result = tool_function(**arguments)
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result
        })
    
    return messages
```

**How This Works:**
`globals()` returns a dictionary of all functions in the global scope. We can look up a function by its name as a string and call it!

**Why This is Better:**
- No need to add an if statement for each new tool
- More maintainable
- More "Pythonic"

**But Remember:**
It's still fundamentally the same thing - we're just mapping tool names to functions and calling them. It's a glorified if statement with fancier syntax!

### Step 11: The Agent Loop

Now we put it all together in the chat function:

```python
def chat(message, history):
    """
    Main chat function with tool calling support.
    
    Args:
        message: Current user message
        history: Previous conversation messages
    
    Returns:
        Agent's response
    """
    # Build messages list
    messages = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": message}
    ]
    
    # Loop until done
    done = False
    while not done:
        # Call OpenAI with tools
        response = openai.chat.completions.create(
            model="gpt-4-mini",
            messages=messages,
            tools=tools  # Our tool descriptions!
        )
        
        # Check what the LLM wants to do
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == "tool_calls":
            # LLM wants to use tools
            tool_calls = response.choices[0].message.tool_calls
            
            # Add LLM's tool request to messages
            messages.append(response.choices[0].message)
            
            # Execute the tools
            tool_results = handle_tool_calls(tool_calls)
            
            # Add results to messages
            messages.extend(tool_results)
            
            # Loop back to call LLM again with results
        else:
            # LLM is done, return the response
            done = True
            return response.choices[0].message.content
```

**Understanding the Loop:**

**Step 1: Build Messages**
Start with system prompt, add history, add current message.

**Step 2: Enter Loop**
We'll keep looping until the LLM is done.

**Step 3: Call OpenAI**
Note we pass `tools=tools` - this is how the LLM knows what tools are available!

**Step 4: Check Finish Reason**
- `"tool_calls"`: LLM wants to use a tool
- `"stop"`: LLM is done, has final answer

**Step 5a: If Tool Calls**
- Add the LLM's message (including tool requests) to messages
- Execute the tools
- Add tool results to messages
- Loop back to Step 3 (call LLM again)

**Step 5b: If Done**
- Return the final response to the user

**Why This Works:**

**First Call:**
User asks a question. If the LLM can answer directly, it does. If it needs a tool, it requests it.

**Second Call:**
We've run the tool and added the results. The LLM now has the information it needs and can generate the final response.

**Multiple Tools:**
If the LLM needs multiple tools, it might request them one at a time, looping several times.

**Example Flow:**

```
User: "Who's your favorite musician?"

Call 1:
LLM: "I don't know" + requests record_unknown_question tool
We: Execute tool, add result to messages

Call 2:
LLM: "I appreciate a variety of music. If you have a favorite, I'd love to hear about it!"
We: Return this to user
```

### Step 12: Testing the Agent

Let's test our agent with different scenarios:

**Test 1: Simple Question**
```
User: "What's your current job?"
Agent: "I'm currently the co-founder and CTO of Nebula..."
```
No tools needed - answered directly from resources.

**Test 2: Unknown Question**
```
User: "Who's your favorite musician?"
Agent: "I'm not sure about my favorite musician..."
[Phone buzzes with notification!]
```
Tool was called to record the unknown question.

**Test 3: User Wants to Connect**
```
User: "I'd like to get in touch"
Agent: "That's great! Please share your email address."
User: "I'm at john@example.com"
Agent: "Thank you! I've recorded your interest..."
[Phone buzzes with notification!]
```
Tool was called to record user details.

**What's Happening:**
The LLM is making intelligent decisions about when to use tools based on the conversation context. You didn't explicitly tell it "if the user says X, do Y" - it figured it out!

### Step 13: Adding an Evaluator (Optional)

Remember the evaluator-optimizer pattern from Day 17? We can add that here too!

**Why Add an Evaluator?**
- Ensures responses are professional
- Catches errors before users see them
- Improves overall quality
- Production-ready reliability

**The Evaluator:**

```python
from pydantic import BaseModel

class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str

def evaluate(reply, message, history):
    """
    Evaluate if a response is acceptable.
    
    Returns:
        Evaluation object with is_acceptable and feedback
    """
    evaluator_prompt = f"""
    You are an evaluator. Decide if this response is acceptable.
    
    The agent should be professional and engaging.
    
    Original question: {message}
    Agent's response: {reply}
    
    Evaluate the response.
    """
    
    response = gemini.beta.chat.completions.parse(
        model="gemini-2.0-flash",
        messages=[{"role": "user", "content": evaluator_prompt}],
        response_format=Evaluation
    )
    
    return response.choices[0].message.parsed
```

**Structured Outputs:**
Notice we're using `response_format=Evaluation`. This is called "structured outputs" - the LLM must respond with an object matching our Pydantic model.

**Pydantic:**
Pydantic is a Python library for data validation. We define a class that describes the structure we want:
- `is_acceptable`: Boolean (True/False)
- `feedback`: String

The LLM's response will be parsed into this structure automatically.

**Connection to Tools:**
Tool calling IS structured outputs! When the LLM responds with tool calls, it's using the same mechanism - returning structured data instead of free-form text.

**Integrating the Evaluator:**

```python
def chat_with_evaluation(message, history):
    # ... build messages ...
    
    # Get initial response
    response = # ... call LLM ...
    reply = response.choices[0].message.content
    
    # Evaluate
    evaluation = evaluate(reply, message, history)
    
    if evaluation.is_acceptable:
        return reply
    else:
        # Regenerate with feedback
        return regenerate_with_feedback(reply, evaluation.feedback, message, history)
```

This adds a quality control layer to ensure all responses meet your standards.

### Step 14: Deployment to Hugging Face Spaces

Now let's make this live on the internet!

**What is Hugging Face Spaces?**
Hugging Face Spaces is a free platform for hosting machine learning applications. It's perfect for Gradio apps and incredibly easy to use.

**Prerequisites:**
1. Create a Hugging Face account at https://huggingface.co
2. Have your app working locally
3. Have your API keys ready

**Deployment Process:**

**Step 1: Navigate to Your Project**
```bash
cd foundations
```

**Step 2: Run Gradio Deploy**
```bash
gradio deploy
```

**Step 3: Answer Questions**

The command will ask you several questions:

**Q: "Enter a title for the app"**
```
Career Conversation
```

**Q: "Enter Gradio app file"**
```
app.py
```
(Just press Enter if your file is named app.py)

**Q: "Enter spaces hardware"**
```
cpu-basic
```
This is the free tier!

**Q: "Any spaces secrets?"**
```
y
```
We need to add our API keys.

**Q: "Enter the secret name"**
```
OPENAI_API_KEY
```

**Q: "Enter the secret value"**
```
[paste your actual OpenAI API key]
```

**Repeat for other secrets:**
- `PUSHOVER_USER`
- `PUSHOVER_TOKEN`

**Q: "Create a GitHub action?"**
```
n
```
Not needed for now.

**Step 4: Wait for Deployment**
The app will be uploaded and deployed. You'll get a URL like:
```
https://huggingface.co/spaces/YOUR_USERNAME/career-conversation
```

**Step 5: Test It!**
Visit the URL and test your deployed app. It's now live on the internet!

**Important Notes:**

**Privacy:**
You can make your space private if you don't want others using your API key. Go to your space settings and change visibility.

**Costs:**
The free tier is sufficient for personal use. If your app goes viral, you might need to upgrade, but OpenAI also has usage limits that protect you.

**Updates:**
To update your app, just run `gradio deploy` again. It will update the existing space.

### Step 15: Embedding in Your Website

Want to put this on your personal website? Hugging Face makes it easy!

**Get the Embed Code:**
1. Go to your space on Hugging Face
2. Click the "Embed" button
3. Copy the iframe code

**Add to Your Website:**
```html
<h2>Chat with Me</h2>
<p>Ask me about my career and experience!</p>

<iframe
  src="https://huggingface.co/spaces/YOUR_USERNAME/career-conversation"
  width="100%"
  height="600px"
  frameborder="0"
></iframe>
```

**Styling:**
You can customize the iframe with CSS to match your website's design.

**Result:**
Visitors to your website can now chat with your AI career assistant! It looks native to your site but is actually hosted on Hugging Face.

## Part 5: Understanding the Modern Agent Loop

### The 2026 Definition

As of 2026, there's a solidifying consensus on what an "agent" means:

**An LLM equipped with tools, running in a loop to achieve a goal.**

Let's break this down:

**1. LLM:**
The intelligence - the language model that makes decisions.

**2. Tools:**
The capabilities - what the LLM can actually do.

**3. Loop:**
Iterative execution - keeps going until the goal is achieved.

**4. Goal:**
A specific objective to accomplish.

### The Experience vs Reality

**What It Feels Like:**
```
You → Call agent → [Agent is off doing its thing...] → Result
```

You make one call, and the agent works autonomously until it's done.

**What Actually Happens:**
```
You → Call LLM → Response: "Use tool X" → You run tool X → 
Call LLM again → Response: "Use tool Y" → You run tool Y →
Call LLM again → Response: "Here's the answer" → Done
```

Your code is orchestrating everything, but it creates the illusion of autonomy.

### A Simple Agent Loop Example

Let's build a to-do list agent to make this concrete:

**The Tools:**

```python
todos = []
completed = []

def create_todos(descriptions):
    """Add items to the to-do list"""
    for desc in descriptions:
        todos.append(desc)
        completed.append(False)
    return get_todo_report()

def mark_completed(index, notes):
    """Mark a to-do item as complete"""
    completed[index] = True
    return get_todo_report()

def get_todo_report():
    """Get a formatted report of all to-dos"""
    report = ""
    for i, (todo, done) in enumerate(zip(todos, completed)):
        status = "✓" if done else "○"
        report += f"{status} {i+1}. {todo}\n"
    return report
```

**The Problem:**
```
"A train leaves Boston at 2pm traveling 60mph. Another train leaves 
New York at 3pm traveling 80mph toward Boston. When do they meet?"
```

**What the Agent Does:**
1. Creates to-dos: "Interpret problem", "Set up variables", "Compute", "Format answer"
2. Works through each to-do, marking complete as it goes
3. Provides the final answer

**The Code:**

```python
def agent_loop(messages):
    done = False
    while not done:
        response = openai.chat.completions.create(
            model="gpt-4-mini",
            messages=messages,
            tools=tools
        )
        
        finish_reason = response.choices[0].finish_reason
        
        if finish_reason == "tool_calls":
            # Execute tools and loop
            tool_results = handle_tool_calls(response.choices[0].message.tool_calls)
            messages.append(response.choices[0].message)
            messages.extend(tool_results)
        else:
            # Done!
            done = True
            print(response.choices[0].message.content)
```

**Running It:**
```python
messages = [
    {"role": "system", "content": "You have to-do tools. Use them to plan and solve problems."},
    {"role": "user", "content": "A train leaves Boston at 2pm..."}
]

agent_loop(messages)
```

**Output:**
```
○ 1. Interpret problem
○ 2. Set up variables
○ 3. Compute meeting time
○ 4. Format answer

✓ 1. Interpret problem
○ 2. Set up variables
○ 3. Compute meeting time
○ 4. Format answer

✓ 1. Interpret problem
✓ 2. Set up variables
○ 3. Compute meeting time
○ 4. Format answer

✓ 1. Interpret problem
✓ 2. Set up variables
✓ 3. Compute meeting time
○ 4. Format answer

✓ 1. Interpret problem
✓ 2. Set up variables
✓ 3. Compute meeting time
✓ 4. Format answer

The trains meet at 4:48pm, 48 miles from Boston.
```

**What Just Happened?**
The agent:
- Planned its own approach
- Created a to-do list
- Worked through each item systematically
- Provided the final answer

All from a simple loop with basic tools!

### Why This Works So Well

**Breaking Down Problems:**
Instead of trying to solve everything at once, the agent breaks it into steps. This improves accuracy.

**Iterative Refinement:**
The agent can check its work, adjust its approach, and refine its solution.

**Tool Use:**
By actually using tools (even simple ones like to-do lists), the agent can organize its thinking and track progress.

**Emergent Behavior:**
The combination of LLM + tools + loop creates behavior that seems remarkably intelligent, even though it's "just" next-token prediction.

### The Key Insight

Remember: It's still just an LLM generating likely next tokens. But by:
- Giving it tools to use
- Running it in a loop
- Letting it see the results of its actions

We create a system that can solve complex problems in a way that feels genuinely intelligent and autonomous.

## Part 6: Commercial Applications and Best Practices

### Real-World Use Cases

**Customer Support:**
- Answer questions using knowledge base (resources)
- Check order status (tools)
- Book appointments (tools)
- Process returns (tools)
- Escalate to humans when needed (tools)

**Personal Assistants:**
- Manage calendar (tools)
- Send emails (tools)
- Research topics (tools + resources)
- Track tasks (tools)
- Set reminders (tools)

**Business Automation:**
- Process documents (resources)
- Update databases (tools)
- Generate reports (resources + tools)
- Notify stakeholders (tools)
- Monitor systems (tools)

**Recruitment:**
- Screen candidates (resources)
- Schedule interviews (tools)
- Answer candidate questions (resources)
- Track applications (tools)

### Best Practices

**Start Simple:**
- Begin with direct API calls (no framework)
- Understand the fundamentals
- Add complexity only as needed

**Clear Prompts:**
- Be explicit about tool usage
- Describe when to use each tool
- Repetition helps! Say important things multiple times

**Tool Design:**
- Keep tools focused and simple
- Each tool should do one thing well
- Clear, descriptive names
- Validate inputs
- Return useful information

**Error Handling:**
- Tools can fail - handle gracefully
- LLMs can make mistakes - validate outputs
- Network issues happen - retry logic
- Set maximum iterations to prevent infinite loops

**Cost Management:**
- Monitor token usage
- Set spending limits
- Use cheaper models where appropriate
- Cache responses when possible

**Security:**
- Validate all tool inputs
- Limit tool capabilities
- Don't expose sensitive functions
- Audit tool usage
- Use environment variables for secrets

**Testing:**
- Test each tool independently
- Test with various prompts
- Check edge cases
- Monitor in production
- Gather user feedback

### Exercises and Next Steps

**Exercise 1: Improve Your Career Agent**
- Add more resources (projects, achievements)
- Implement RAG for better context
- Add more tools (database, calendar)
- Integrate the evaluator
- Improve the UI

**Exercise 2: Build Your Own Agent Loop**
- Start a fresh notebook
- Create simple tools
- Implement the loop from scratch
- Test with different problems
- Share your results!

**Exercise 3: Deploy and Share**
- Deploy your agent to Hugging Face
- Embed it in your website
- Share on LinkedIn
- Get feedback
- Iterate based on responses

## Conclusion

Today you've learned how to build real AI agents from first principles. You understand:

**Resources:**
How to enhance LLM capabilities with context and information.

**Tools:**
How to give LLMs the power to take actions, and that it's "just" JSON and if statements.

**Agent Loops:**
How LLM + tools + loop = agent, and why this approach is so effective.

**Practical Skills:**
You've built a deployable career agent that you can actually use professionally.

**Foundation for Frameworks:**
Next week, when we start using OpenAI Agents SDK, you'll understand exactly what it's doing under the hood.

### Key Takeaways

1. **No Magic:** Tool calling is simpler than it seems - it's JSON and if statements.

2. **Resources are Powerful:** Adding context transforms generic LLMs into domain experts.

3. **Agent Loops Work:** The combination of LLM + tools + loop creates remarkably effective systems.

4. **Start Simple:** Understanding the fundamentals makes you a better developer when using frameworks.

5. **It's Practical:** You've built something you can actually deploy and use professionally.

### Looking Ahead

**Next Week:**
OpenAI Agents SDK - our first framework! You'll see how it automates all the boilerplate we wrote today.

**Remember:**
- Theory guides practice
- Fundamentals matter
- Simple is often better
- Keep building and experimenting!

**The journey continues!**

Welcome to the world of building real AI agents. You now have the skills to create agents that can take actions, interact with the world, and solve real problems. This is just the beginning!

