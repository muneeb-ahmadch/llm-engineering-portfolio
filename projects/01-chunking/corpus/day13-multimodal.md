# Day 13: Building UIs, Tools & Multimodal AI
## Detailed Lecture Notes

---

## Introduction

Welcome to Day 13! Building on what we learned in Day 12 about the Chat Completions API, tokens, and memory, today we're going to put everything together to build real applications. We'll cover frameworks that make working with LLMs easier, learn how to create user interfaces without any frontend knowledge, understand how tool calling works (spoiler: it's simpler than you think!), and create a multimodal AI assistant that can generate images and speak.

---

## Part 1: LLM Frameworks

### Why Use Frameworks?

When you're building real applications, you often need to:
- Switch between different LLM providers easily
- Track costs across many API calls
- Use a consistent interface regardless of which model you're calling

That's where LLM frameworks come in. Think of them as helpful wrappers that sit between your code and the various LLM APIs.

### LangChain

LangChain is probably the most famous LLM framework. It's powerful and has a lot of features, but it's also quite heavyweight. There's a lot to learn, and it can feel overwhelming at first.

**Basic LangChain example:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
response = llm.invoke("Tell me a joke")
print(response.content)
```

LangChain is excellent for complex applications where you need its advanced features like chains, agents, and memory management. We'll explore it more in later sessions.

### LiteLLM

On the opposite end, LiteLLM is a lightweight framework that does one thing really well: it gives you a simple, consistent way to call any LLM provider.

**Basic LiteLLM example:**
```python
from litellm import completion

response = completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a joke"}]
)
print(response.choices[0].message.content)
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

**Example scenario:** You have a customer service bot that always includes your company's 10-page policy document in the system prompt. Instead of processing that document from scratch every time, caching lets the provider remember it.

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
    
    # Claude responds
    claude_response = call_claude(claude_messages)
    
    # Add Claude's response to GPT's messages as if a user said it
    gpt_messages.append({"role": "user", "content": claude_response})
```

The result? Hilarious conversations where GPT is snarky and Claude keeps trying to make peace!

### Three-Way Conversations

What if you want three LLMs talking? The user/assistant format doesn't work well here. Instead, use a different approach:

**System prompt:**
```
You are Alex, a chatbot who is very argumentative. 
You're in a conversation with Blake and Charlie.
```

**User prompt:**
```
The conversation so far is:
Blake: "Hello everyone!"
Charlie: "Hi Blake, nice to see you"

Now respond as Alex.
```

This technique of putting the full context in a single user prompt is powerful and works for many complex scenarios beyond just conversations.

---

## Part 4: Building UIs with Gradio

### What is Gradio?

Gradio is a Python library that lets you create web interfaces without knowing any frontend code. It was created by a company now owned by Hugging Face, and it's incredibly popular in the data science community.

### Your First Gradio App

```python
import gradio as gr

def shout(text):
    return text.upper()

gr.Interface(fn=shout, inputs="text", outputs="text").launch()
```

That's it! Run this code and a web UI appears where users can type text and see it converted to uppercase.

### How Gradio Works (Under the Hood)

When you call `.launch()`, three things happen:

1. **Frontend Generation:** Gradio converts your Python description into a Svelte frontend (which compiles to JavaScript)

2. **Server Startup:** It starts a Starlette web server on your machine (usually port 7860)

3. **Route Creation:** It creates API endpoints for your callback functions

When someone clicks "Submit" in the UI, the frontend calls the backend route, which runs your Python function, and the result gets displayed. Simple!

### Gradio Interface Types

**1. gr.Interface - Simple Input/Output**
```python
gr.Interface(
    fn=my_function,
    inputs=gr.Textbox(label="Your Message"),
    outputs=gr.Markdown(),
    title="My App"
).launch()
```

**2. gr.ChatInterface - Chat Style**
```python
def chat(message, history):
    # Your LLM call here
    return response

gr.ChatInterface(fn=chat, type="messages").launch()
```

This creates a complete chat UI automatically! The `history` parameter receives all previous messages in OpenAI format.

**3. gr.Blocks - Custom Layouts**
```python
with gr.Blocks() as app:
    with gr.Row():
        chatbot = gr.Chatbot()
        image = gr.Image()
    with gr.Row():
        textbox = gr.Textbox()
    
    textbox.submit(fn=respond, inputs=[chatbot], outputs=[chatbot, image])

app.launch()
```

### Useful Gradio Features

**Streaming responses:**
```python
def chat(message, history):
    response = ""
    for chunk in stream_from_llm(message):
        response += chunk
        yield response  # Use yield instead of return
```

Gradio automatically detects generators and updates the UI as chunks arrive!

**Sharing your app:**
```python
gr.Interface(...).launch(share=True)
```
This creates a public URL (via Gradio's servers) that anyone can access for 72 hours.

**Adding authentication:**
```python
gr.Interface(...).launch(auth=("username", "password"))
```

---

## Part 5: Building AI Assistants

### The Key Ingredients

A good AI assistant has:
1. **Persona** - Set through the system prompt
2. **Memory** - By including conversation history (as we learned in Day 12)
3. **Expertise** - Through context injection

### Prompting Techniques

**One-Shot Prompting:**
Give one example of how to respond:
```
You are a helpful clothes store assistant.
Example: If the customer says "I'm looking for a hat", you could reply 
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
4. **You** call the LLM again with the results
5. The LLM gives a final response

It's just a structured conversation! The LLM generates tokens that say "call this function", and your code interprets those tokens.

### Defining Tools

Tools are described in JSON schema format:
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_ticket_price",
        "description": "Get the price of a return ticket to a destination city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The destination city"
                }
            },
            "required": ["city"]
        }
    }
}]
```

Yes, it's verbose. But LLMs have been trained on this format, so they know exactly what to do with it.

### Using Tools in Practice

```python
# Pass tools when calling the API
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools
)

# Check if the LLM wants to use a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    
    # Actually run your function
    if tool_call.function.name == "get_ticket_price":
        city = json.loads(tool_call.function.arguments)["city"]
        result = get_ticket_price(city)
    
    # Add the tool response to messages
    messages.append(response.choices[0].message)  # The tool request
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": result
    })
    
    # Call the LLM again with the results
    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools
    )
```

### Handling Multiple Tool Calls

An LLM might want to call multiple tools at once (e.g., "Compare prices to London and Paris"). Use a loop:

```python
while response.choices[0].finish_reason == "tool_calls":
    for tool_call in response.choices[0].message.tool_calls:
        # Handle each tool call
        result = handle_tool(tool_call)
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
```

---

## Part 7: Agentic AI Concepts

### What is an Agent?

There are many definitions, but two main ones:

1. **Workflow Control:** An LLM that decides what happens next, orchestrating different actions
2. **Goal-Oriented Loop:** An LLM that repeatedly uses tools until it achieves a goal

Think of Claude Code or Cursor's agent mode - they keep working, using tools, checking results, and iterating until the task is done.

### Characteristics of Agents

- **Memory:** Remembering context across interactions
- **Planning:** Breaking down tasks into steps
- **Autonomy:** Making decisions about what to do next
- **Tool Use:** Taking actions beyond generating text

What we built today with tools is a stepping stone toward full agentic systems!

---

## Part 8: Multimodal AI

### Beyond Text

Modern AI can work with multiple modalities:
- **Text** (what we've been doing)
- **Images** (generation and understanding)
- **Audio** (speech-to-text, text-to-speech)
- **Video** (coming soon to more models)

### Image Generation with DALL-E

```python
response = client.images.generate(
    model="dall-e-3",
    prompt="A vibrant pop art image of Tokyo with tourist landmarks",
    size="1024x1024",
    response_format="url"
)

image_url = response.data[0].url
```

**Note:** Image generation costs about $0.04 per image - much more than text generation!

**Prompt tip:** Be specific about style, content, and mood. "A vacation in Paris" is okay, but "A vibrant pop art illustration of Paris showing the Eiffel Tower, cafes, and the Seine River" is better.

### Text-to-Speech

```python
response = client.audio.speech.create(
    model="gpt-4o-mini-tts",
    voice="onyx",  # Options: alloy, echo, fable, nova, onyx, shimmer
    input="Hello! Welcome to our airline booking service."
)

# Save or play the audio
audio_content = response.content
```

### Building a Multimodal Assistant

Now we can combine everything:
1. **Chat interface** with Gradio
2. **Tool calling** for database lookups
3. **Image generation** for visual content
4. **Text-to-speech** for audio responses

```python
def chat(history):
    # Call LLM with tools
    response, cities = call_llm_with_tools(history)
    
    # Generate image if a city was mentioned
    image = None
    if cities:
        image = generate_city_image(cities[0])
    
    # Generate audio of the response
    audio = text_to_speech(response)
    
    return history + [response], audio, image
```

With Gradio Blocks, you can create a custom UI that shows the chat, plays audio, and displays images - all from one Python file!

---

## Summary

Today we covered a lot of ground:

1. **LLM Frameworks** - LangChain for complex needs, LiteLLM for simplicity
2. **Prompt Caching** - Save money on repeated prompts
3. **Gradio** - Build UIs in minutes with just Python
4. **AI Assistants** - Use prompting techniques for consistent behavior
5. **Tool Calling** - Extend LLM capabilities (remember: no magic!)
6. **Multimodal AI** - Images and audio add richness to applications

The most important takeaway: **All of this is simpler than it seems.** LLMs just generate tokens. Everything else - tools, memory, agents - is clever engineering on top of that basic capability.

---

## Practice Exercises

1. Create a Gradio chatbot with a custom persona for your use case
2. Add a tool that looks up information relevant to your domain
3. Experiment with different DALL-E prompts to understand prompt engineering for images
4. Build a multimodal demo combining chat, images, and/or audio

---

## Additional Resources

- **Gradio Documentation:** https://www.gradio.app/docs
- **LiteLLM Documentation:** https://docs.litellm.ai/
- **OpenAI Function Calling Guide:** https://platform.openai.com/docs/guides/function-calling
- **DALL-E Prompt Guide:** https://help.openai.com/en/articles/6516417-dall-e-prompt-guide

---

## What's Next?

In upcoming sessions, we'll dive into:
- Open-source models with Hugging Face
- Running models on GPUs with Google Colab
- Understanding tokenizers and transformers at a deeper level

See you next time!

