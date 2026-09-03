# Day 15 Lecture Notes: Building RAG with LangChain & Chroma

## Introduction

Welcome to Day 15! Today is an exciting day - we're going to build a complete, working RAG (Retrieval Augmented Generation) system. Yesterday (Day 14), we learned the theory behind RAG and vector embeddings. Today, we put it all into practice.

By the end of today, you'll have:
- A working vector database with your documents
- A complete RAG pipeline
- A Gradio web interface for your RAG system
- The ability to visualize vectors in 2D and 3D

Let's get started!

## Quick Recap: The Big Idea Behind RAG

Before we dive in, let's quickly recap what we learned yesterday:

**The RAG Flow**:
1. **User asks a question** - "Who is Avery Lancaster?"
2. **Encode the question** - Turn it into a vector using an encoder LLM
3. **Search vector database** - Find chunks with similar vectors
4. **Retrieve text** - Get the actual text of those similar chunks
5. **Augment prompt** - Add retrieved context to the prompt
6. **Generate response** - LLM answers using both its knowledge and the retrieved context

**Why RAG?**
- LLMs have knowledge cutoffs
- Can't access private/proprietary data
- Context windows have limits
- RAG solves all these problems!

## Part 1: Introducing LangChain

### What is LangChain?

LangChain is an open-source framework that makes it easier to build applications with Large Language Models. It was launched in October 2022 by Harrison Chase, who later formed a company around it.

**Key Concept**: LangChain is an "abstraction layer" - it provides a common interface for working with different LLMs and helps you chain them together to accomplish complex tasks.

**Recent Update**: Version 1.0 was released in October 2025, representing a significant overhaul with repackaged modules and improved organization.

**Website**: https://python.langchain.com/

### The Pros of LangChain

**1. Quick to Market**
LangChain dramatically speeds up development:
- Pre-built components for common tasks (RAG, agents, summarization)
- Easy integration with multiple LLM providers
- Robust tooling out of the box
- Reduces boilerplate code

**Example**: Building a RAG system might take 100+ lines of custom code, but with LangChain, it's just 5-10 lines.

**2. Enterprise Adoption**
- Extremely popular since 2023
- Widely used in production systems
- Good for your resume and career
- Large community and ecosystem

**3. Comprehensive Tooling**
- Document loaders for various formats
- Text splitters for chunking
- Vector store integrations
- Retrieval mechanisms
- Agent frameworks

### The Cons of LangChain

**1. Less Needed Than Before**

When LangChain launched in 2022-2023, calling different LLM providers required completely different code. LangChain unified this.

**Then**: Different APIs for OpenAI, Anthropic, Google
**Now**: Everyone has OpenAI-compatible endpoints

You can now easily switch between providers by just changing the base URL:

```python
# OpenAI
client = OpenAI(api_key=key)

# Anthropic (OpenAI-compatible)
client = OpenAI(base_url="https://api.anthropic.com/v1", api_key=key)

# Gemini (OpenAI-compatible)
client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=key)
```

**Result**: Less need for an abstraction layer

**2. Heavyweight Framework**

LangChain has grown significantly:
- **Multiple packages**: `langchain-openai`, `langchain-community`, `langchain-chroma`, `langchain-huggingface`
- **Custom terminology**: HumanMessage, AIMessage, SystemMessage (vs simple dicts)
- **Own language**: LCEL (LangChain Expression Language)
- **Steep learning curve**: Lots of concepts to learn

**Alternatives**:
- **LiteLLM**: Very lightweight, minimal learning curve
- **Direct API calls**: Sometimes simpler for basic use cases

**3. Legacy Feel**

Some aspects feel dated:
- Message format (HumanMessage objects vs OpenAI's simple list of dicts)
- Heavy dependencies
- Terminology that's specific to LangChain

### The LangChain Ecosystem

LangChain is now part of a larger ecosystem:

**LangChain**: Core framework for LLM applications
**LangGraph**: For building agent workflows with dependency graphs
**LangSmith**: Observability and monitoring platform
**LangServe**: Deployment tools

### Our Verdict

LangChain is a powerful tool with both advantages and disadvantages. You'll experience both firsthand today:
- **The Good**: Building a RAG pipeline in just a few lines
- **The Rough Edges**: Multiple packages, custom terminology, learning curve

It's valuable to know, widely used in industry, but not always necessary for every project.

## Part 2: Document Chunking

### Why Do We Need Chunks?

Imagine you have documents containing:
- Employee HR records
- Product descriptions
- Contract terms
- Company policies

When a user asks "Who is Avery Lancaster?", the answer is in ONE specific part of ONE document, not the entire document collection.

**Problem with Whole Documents**:
- If we create one vector per entire document, it represents ALL the information in that document
- User questions are specific and focused
- Matching a specific question to a general document vector is inefficient

**Solution: Chunking**:
- Break documents into smaller, focused pieces
- Each chunk covers a specific topic or section
- Better chance of matching question to relevant chunk
- More precise retrieval

### Example

**Original Document** (Employee Record):
```
Name: Avery Lancaster
Position: CEO
Department: Executive
Salary: $250,000
Start Date: 2020-01-15
Education: MBA from Stanford
Previous Experience: VP at TechCorp
Skills: Leadership, Strategy, Finance
```

**Chunked** (with overlap):
```
Chunk 1: "Name: Avery Lancaster, Position: CEO, Department: Executive"
Chunk 2: "Position: CEO, Department: Executive, Salary: $250,000, Start Date: 2020-01-15"
Chunk 3: "Start Date: 2020-01-15, Education: MBA from Stanford, Previous Experience: VP at TechCorp"
```

Now a question about Avery's education will match Chunk 3 specifically!

### Text Splitters in LangChain

LangChain provides several text splitters:

**1. CharacterTextSplitter**
- Simplest approach
- Splits by character count
- No intelligence about content structure
- Use case: When structure doesn't matter

**2. RecursiveCharacterTextSplitter** (Most Common)
- Tries to split at natural boundaries
- Hierarchy of splitting:
  1. First: Double newlines (paragraph breaks)
  2. Then: Single newlines (line breaks)
  3. Then: Sentences
  4. Finally: Words
- Preserves context better
- Use case: Most documents

**3. Others**
- TokenTextSplitter: Splits by token count
- MarkdownTextSplitter: Respects markdown structure
- PythonCodeTextSplitter: Understands Python syntax

### Key Parameters

**chunk_size**: Target size in characters
- Too small: Fragments lose context
- Too large: Too much irrelevant information
- Typical: 500-2000 characters
- Our example: 1000 characters

**chunk_overlap**: Overlap between chunks
- Prevents splitting answers across boundaries
- Typical: 10-20% of chunk size
- Our example: 200 characters (20% of 1000)

### Example Code

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents
chunks = text_splitter.split_documents(documents)

print(f"Split {len(documents)} documents into {len(chunks)} chunks")
```

### Why Overlap Matters

**Without Overlap**:
```
Chunk 1: "...Avery Lancaster has extensive experience in"
Chunk 2: "insurance technology and has led the company..."
```
Question: "What experience does Avery have?" - Might miss the complete answer!

**With Overlap**:
```
Chunk 1: "...Avery Lancaster has extensive experience in insurance"
Chunk 2: "experience in insurance technology and has led the company..."
```
Question: "What experience does Avery have?" - Chunk 2 has complete context!

### Chunking is Experimental

**Important**: There's no one "right" way to chunk documents. It's:
- **Empirical**: Based on testing and measurement
- **Domain-specific**: Different for legal docs vs chat logs
- **Iterative**: Requires experimentation

**Best Practice**: Try different strategies and measure which works best for your use case.

## Part 3: Encoder Models (Embeddings)

### Two Types of LLMs Revisited

We learned yesterday that there are two fundamentally different types of LLMs:

**Type 1: Autoregressive LLMs**
- What we've used so far: GPT, Claude, Gemini
- **Job**: Predict the next token
- **Input**: Text sequence
- **Output**: Next token (repeated to generate full response)
- **Use**: Text generation, chat, coding

**Type 2: Encoder LLMs**
- New focus today: BERT, OpenAI Embeddings, Sentence Transformers
- **Job**: Represent the entire input as a vector
- **Input**: Text sequence
- **Output**: One vector (list of numbers)
- **Use**: Classification, semantic search, embeddings

### What is a Vector Embedding?

A vector embedding is a list of numbers that represents the meaning of text.

**Example**:
```
Text: "Avery Lancaster is the CEO of InsureElm"
Vector: [0.23, -0.15, 0.67, 0.02, ..., 0.45]  # 384 numbers
```

**Key Property**: Similar meanings → Similar vectors

```
"CEO of InsureElm" → [0.23, -0.15, 0.67, ...]
"Chief Executive Officer at InsureElm" → [0.24, -0.14, 0.68, ...]
```

Different words, same meaning, close vectors!

### Popular Encoder Models

**OpenAI Embeddings**:

1. **text-embedding-3-small**
   - Dimensions: 1,536
   - Cost: ~$0.02 per 1M tokens
   - Quality: Very good
   - Speed: Fast
   - Use: Most applications

2. **text-embedding-3-large**
   - Dimensions: 3,072
   - Cost: ~$0.13 per 1M tokens
   - Quality: Best
   - Speed: Slower
   - Use: When quality is critical

**Google Embeddings**:

**gemini-embedding-001**
- Dimensions: 768
- Quality: Good
- Integration: Easy with Google Cloud

**Hugging Face (Open Source)**:

**all-MiniLM-L6-v2** (Most Popular)
- Dimensions: 384
- Cost: Free!
- Quality: Good for open source
- Speed: Very fast
- Use: Development, cost-sensitive applications

**Historical Context**:
- **Word2Vec** (2013): First popular word embeddings
- **BERT** (2018): Google's breakthrough transformer encoder
- **Sentence Transformers** (2019): Optimized for sentence-level embeddings

### Dimensions Explained

**What are dimensions?**
- The number of numbers in the vector
- More dimensions = more "degrees of freedom" to represent meaning
- Think of it like coordinates in space

**Examples**:
- 2D: (x, y) - A point on a map
- 3D: (x, y, z) - A point in space
- 384D: 384 numbers - A point in 384-dimensional space!

**More Dimensions = Better?**
Generally yes, but:
- More dimensions = more computational cost
- More dimensions = more storage
- Diminishing returns after a point
- Quality of the model matters more than dimensions alone

### Critical Distinction: Encoder vs Database

This is crucial to understand:

**Encoder Model (e.g., all-MiniLM-L6-v2)**:
- **Creates** the vectors
- **Determines** the dimensionality
- **Affects** the quality of semantic search
- **This is what you should focus on choosing carefully**

**Vector Database (e.g., Chroma)**:
- **Stores** the vectors
- **Enables** fast retrieval
- **Infrastructure** choice (like choosing MySQL vs PostgreSQL)
- **Less critical** for semantic quality

**Analogy**:
- Encoder = Camera (determines image quality)
- Database = Photo album (stores the photos)

You wouldn't blame your photo album for blurry photos - that's the camera's job!

## Part 4: Vector Databases

### What is a Vector Database?

A vector database is specialized for:
- Storing vectors efficiently
- Finding similar vectors quickly
- Performing "nearest neighbor" searches

**Traditional Database**:
```sql
SELECT * FROM users WHERE name = 'Avery Lancaster'
```
Exact match on text

**Vector Database**:
```python
vector_db.similarity_search(query_vector, k=5)
```
Find 5 most similar vectors

### Popular Vector Databases

**Open Source**:

**1. Chroma** (We'll use today)
- Easy to use
- SQLite-based
- Perfect for development
- Free and open source
- Website: https://www.trychroma.com/

**2. FAISS** (Facebook AI Similarity Search)
- Very fast
- In-memory
- Great for prototyping
- Not a full database (no persistence by default)
- Website: https://github.com/facebookresearch/faiss

**Paid/Managed**:

**1. Pinecone**
- Fully managed
- Scales to billions of vectors
- Enterprise features
- Pay per usage
- Website: https://www.pinecone.io/

**2. Weaviate**
- Open source with managed option
- GraphQL API
- Hybrid search
- Website: https://weaviate.io/

**Mainstream Databases with Vector Support**:

Modern databases now support vectors natively:
- **PostgreSQL** (with pgvector extension)
- **MongoDB** (Atlas Vector Search)
- **Elasticsearch** (dense_vector field type)
- **Redis** (RediSearch with vectors)

**Advantage**: No need for separate vector database if you're already using these!

### Choosing a Vector Database

**For Development/Learning**: Chroma or FAISS
**For Production (Small Scale)**: Chroma, PostgreSQL with pgvector
**For Production (Large Scale)**: Pinecone, Weaviate, or Elasticsearch
**For Existing Infrastructure**: Add vectors to your current database

**Remember**: The encoder choice matters much more than the database choice!

## Part 5: Creating Vectors with LangChain

### Loading Documents

First, we need to load our documents. LangChain provides document loaders:

```python
from langchain_community.document_loaders import DirectoryLoader

# Load all markdown files from a directory
loader = DirectoryLoader(
    "knowledge_base/products/",
    glob="**/*.md"
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")
```

Each document is a LangChain `Document` object with:
- `page_content`: The text content
- `metadata`: Information about the document (source, type, etc.)

### Creating Chunks

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Split documents into chunks
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks from {len(documents)} documents")
```

**Example Output**:
```
Loaded 76 documents
Created 413 chunks from 76 documents
```

### Creating the Vector Store

Now for the magic - creating vectors and storing them in Chroma:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Create encoder
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Create vector store
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vector_db"
)

print("✅ Vector store created!")
```

**What Just Happened?**:
1. Created an encoder (all-MiniLM-L6-v2)
2. For each of 413 chunks:
   - Encoder converted text → 384-dimensional vector
   - Chroma stored the vector + original text
3. Created a searchable database

**That's it!** You now have a working vector database.

### Inspecting the Vector Store

```python
# Get the collection
collection = vector_store._collection

# Check count
print(f"Vectors in database: {collection.count()}")

# Check dimensions
sample = collection.get(limit=1, include=["embeddings"])
dimensions = len(sample["embeddings"][0])
print(f"Vector dimensions: {dimensions}")
```

**Output**:
```
Vectors in database: 413
Vector dimensions: 384
```

### Using Different Encoders

**OpenAI Small**:
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)
# Dimensions: 1,536
```

**OpenAI Large**:
```python
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)
# Dimensions: 3,072
```

**Switching is easy!** Just change the embeddings object.

## Part 6: Visualizing Vectors

### Why Visualize?

Visualizing vectors helps you:
- **Understand** how the encoder groups similar content
- **Spot** outliers and data quality issues
- **Compare** different encoders
- **Debug** retrieval problems
- **Build intuition** about vector spaces

### The Challenge: High Dimensions

Humans can visualize:
- 1D: A line
- 2D: A plane (x, y)
- 3D: Space (x, y, z)

But our vectors have 384, 1536, or 3072 dimensions!

**Solution**: Dimensionality reduction

### t-SNE: The Magic Tool

**t-SNE** = t-distributed Stochastic Neighbor Embedding

**What it does**:
- Takes high-dimensional data (384D, 1536D, etc.)
- Projects it down to 2D or 3D
- Preserves similarity relationships
- Points close in high-D space → Close in 2D/3D
- Points far in high-D space → Far in 2D/3D

**How to use**:
```python
from sklearn.manifold import TSNE
import plotly.express as px

# Get vectors from Chroma
vectors = [...]  # 413 vectors, each 384 dimensions

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# Plot
fig = px.scatter(
    x=vectors_2d[:, 0],
    y=vectors_2d[:, 1],
    color=doc_types,
    hover_data={"text": texts}
)
fig.show()
```

### What You'll See

**With all-MiniLM-L6-v2** (384 dimensions):
- Some clustering by document type
- Some overlap between categories
- Generally decent separation

**With text-embedding-3-small** (1,536 dimensions):
- Much better clustering
- Clearer separation between document types
- Employees clearly separated from products

**With text-embedding-3-large** (3,072 dimensions):
- Excellent clustering
- Very clear separation
- Each document type in its own region
- Minimal overlap

**Key Insight**: Better encoders create better vector spaces!

### 3D Visualization

Even better - we can visualize in 3D:

```python
# Reduce to 3D
tsne_3d = TSNE(n_components=3, random_state=42)
vectors_3d = tsne_3d.fit_transform(vectors)

# 3D scatter plot
fig = px.scatter_3d(
    x=vectors_3d[:, 0],
    y=vectors_3d[:, 1],
    z=vectors_3d[:, 2],
    color=doc_types,
    hover_data={"text": texts}
)
fig.show()
```

**Advantages of 3D**:
- Can rotate to see different angles
- Better understanding of cluster separation
- More intuitive for spatial relationships
- Fun to explore!

### Interpreting Visualizations

**What to look for**:
1. **Clusters**: Do similar documents group together?
2. **Separation**: Are different types well separated?
3. **Outliers**: Any points far from their group?
4. **Overlap**: Where do categories mix? (Often meaningful!)

**Example Insights**:
- Product features cluster near contract features (they discuss similar topics!)
- Employee records mostly separate (different vocabulary)
- Company info appears in multiple clusters (mentioned in many contexts)

## Part 7: Building the Complete RAG Pipeline

### The RAG Components

A complete RAG system needs:
1. **Vector Store**: Chroma with our chunks
2. **Retriever**: Finds relevant chunks
3. **LLM**: Generates responses
4. **Prompt Template**: Structures the context

### Creating the Retriever

```python
# Create retriever from vector store
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Retrieve top 5 chunks
)
```

**What it does**:
- Takes a question
- Converts question to vector (using same encoder!)
- Finds 5 most similar chunk vectors
- Returns the text of those chunks

### The RAG Function

Here's a complete RAG implementation:

```python
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

def answer_question(question, history=[]):
    # 1. Fetch relevant context
    context_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # 2. Build system prompt with context
    system_prompt = f"""You are an expert on InsureElm insurance company.
Use the following context to answer questions accurately.

Context:
{context}

Answer the question based on the context provided."""
    
    # 3. Build messages
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # 4. Get response from LLM
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(messages)
    
    return response.content
```

**That's it!** A complete RAG system in about 20 lines.

### Testing the RAG System

```python
# Test 1: Simple question
answer = answer_question("Who is Avery Lancaster?")
print(answer)
# Output: "Avery Lancaster is the Co-founder and CEO of InsureElm..."

# Test 2: With typo
answer = answer_question("Who is Avry Lancster?")
print(answer)
# Output: "There might be a typo. You may be referring to Avery Lancaster..."

# Test 3: Synonym
answer = answer_question("Who is the CEO?")
print(answer)
# Output: "Avery Lancaster is the CEO of InsureElm..."
```

**Magic!** The vector similarity handles:
- Typos
- Synonyms
- Different phrasings
- Partial names

### Handling Conversation History

LLMs are stateless - they don't remember previous messages. We must pass the full history:

```python
def answer_question(question, history=[]):
    # Fetch context
    context_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Build system prompt
    system_prompt = f"""Context: {context}
    
Answer based on the context."""
    
    # Build messages with history
    messages = [SystemMessage(content=system_prompt)]
    
    # Add conversation history
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))
    
    # Add current question
    messages.append(HumanMessage(content=question))
    
    # Get response
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke(messages)
    
    return response.content
```

**Now it works**:
```
User: "Who is Avery?"
AI: "Avery Lancaster is the CEO..."

User: "What did she do before?"
AI: "Before joining InsureElm, Avery was VP at TechCorp..."
```

### Advanced: Context for Follow-ups

**Problem**: When retrieving context for "What did she do before?", should we:
- A) Only use the latest question?
- B) Combine all user questions?
- C) Use the full conversation?

**Answer**: Depends on use case!

**Option B Implementation**:
```python
def answer_question(question, history=[]):
    # Combine all user questions
    all_questions = [q for q, a in history] + [question]
    combined_question = " ".join(all_questions)
    
    # Fetch context based on combined question
    context_docs = retriever.invoke(combined_question)
    
    # Rest of the code...
```

**Trade-offs**:
- Option A: Fast, but might miss context
- Option B: Better context, but slower
- Option C: Most context, but can be noisy

**Experiment and measure!**

## Part 8: Production RAG Architecture

### Modular Design

For production, separate concerns into modules:

**ingest.py** - Data ingestion pipeline:
```python
def fetch_documents():
    """Load documents from knowledge base"""
    # Load from all directories
    pass

def create_chunks(documents):
    """Split documents into chunks"""
    # Use text splitter
    pass

def create_embeddings(chunks):
    """Create vector store"""
    # Create Chroma database
    pass

if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    create_embeddings(chunks)
    print("✅ Ingestion complete")
```

**answer.py** - Query interface:
```python
def fetch_context(question):
    """Retrieve relevant chunks"""
    return retriever.invoke(question)

def answer_question(question, history=[]):
    """Generate answer with RAG"""
    context = fetch_context(question)
    # Build prompt and call LLM
    return response
```

**app.py** - User interface:
```python
import gradio as gr
from implementation.answer import answer_question

demo = gr.ChatInterface(
    fn=answer_question,
    title="InsureElm AI Assistant"
)

demo.launch()
```

### Benefits of Modular Design

1. **Separation of Concerns**:
   - Ingestion runs independently (cron job, data pipeline)
   - Query interface is stateless
   - UI is swappable

2. **Easy to Test**:
   - Test ingestion separately
   - Test retrieval separately
   - Test generation separately

3. **Swappable Implementations**:
   - Try different encoders
   - Try different chunking strategies
   - Compare approaches

4. **Production Ready**:
   - Ingest can run on schedule
   - Query can be an API
   - UI can be web, mobile, CLI, etc.

## Part 9: Gradio RAG Application

### Creating the Interface

```python
import gradio as gr
from implementation.answer import answer_question

demo = gr.ChatInterface(
    fn=answer_question,
    title="InsureElm RAG Assistant",
    description="Ask questions about InsureElm employees and products",
    examples=[
        "Who is Avery Lancaster?",
        "What products does InsureElm offer?",
        "Tell me about the CEO",
    ]
)

demo.launch(share=False)
```

### Features

**Conversation History**:
- Maintains context across questions
- Handles follow-up questions
- Displays full conversation

**Error Handling**:
- Graceful handling of typos
- Suggests corrections
- Handles edge cases

**User Experience**:
- Clean interface
- Fast responses
- Shows thinking (optional)

### Advanced Features

**Show Source Documents**:
```python
def answer_with_sources(question, history):
    # Get context
    context_docs = retriever.invoke(question)
    
    # Generate answer
    answer = generate_answer(question, context_docs, history)
    
    # Format sources
    sources = "\n\n**Sources:**\n"
    for i, doc in enumerate(context_docs, 1):
        sources += f"{i}. {doc.metadata['source']}\n"
    
    return answer + sources
```

**Confidence Scores**:
```python
# Get similarity scores
results = vector_store.similarity_search_with_score(question, k=5)

# Check if confident
if results[0][1] < 0.5:  # Low similarity
    return "I'm not confident I have information about that."
```

## Part 10: Comparing Encoders

### The Experiment

We tested three encoders on the same 413 chunks:

**1. all-MiniLM-L6-v2** (384 dimensions)
- Free, open source
- Fast
- Decent quality

**2. text-embedding-3-small** (1,536 dimensions)
- Paid ($0.02/1M tokens)
- Good quality
- 4x more dimensions

**3. text-embedding-3-large** (3,072 dimensions)
- Paid ($0.13/1M tokens)
- Best quality
- 8x more dimensions

### Results

**Visualization Comparison**:

**all-MiniLM-L6-v2**:
- Some clustering visible
- Overlap between categories
- Employees mostly together
- Products and contracts mixed

**text-embedding-3-small**:
- Much clearer clustering
- Better separation
- Employees well separated
- Products distinct from contracts

**text-embedding-3-large**:
- Excellent clustering
- Very clear separation
- Each type in its own region
- Minimal overlap

**Conclusion**: More sophisticated encoders create better vector spaces!

### Practical Implications

**For Development**:
- Start with all-MiniLM-L6-v2 (free, fast)
- Test your chunking strategy
- Validate your approach

**For Production**:
- Upgrade to OpenAI embeddings
- Better retrieval quality
- Worth the cost for most applications

**For Critical Applications**:
- Use text-embedding-3-large
- Best possible retrieval
- Essential for high-stakes decisions

### Cost Considerations

**Example**: 413 chunks, average 500 tokens each = ~206,500 tokens

**Costs**:
- all-MiniLM-L6-v2: **$0** (free!)
- text-embedding-3-small: **$0.004** (less than a penny)
- text-embedding-3-large: **$0.027** (3 cents)

**For most applications, the quality improvement is worth the minimal cost!**

## Key Takeaways

### What You've Learned Today

1. **LangChain Fundamentals**
   - Pros: Quick development, robust tooling, enterprise adoption
   - Cons: Heavyweight, less needed now, steep learning curve
   - Use when: Building complex LLM applications quickly

2. **Document Chunking**
   - Break documents into focused pieces
   - Use RecursiveCharacterTextSplitter
   - Typical: 1000 chars with 200 char overlap
   - Experiment to find what works!

3. **Encoder Models**
   - Create vectors from text
   - Popular: all-MiniLM-L6-v2, OpenAI embeddings
   - More dimensions generally better
   - This is what affects quality!

4. **Vector Databases**
   - Store and search vectors
   - Popular: Chroma, FAISS, Pinecone
   - Infrastructure choice
   - Less critical than encoder choice

5. **Visualization**
   - Use t-SNE for dimensionality reduction
   - 2D and 3D visualizations
   - Understand encoder behavior
   - Spot patterns and outliers

6. **Complete RAG Pipeline**
   - Retriever + LLM + Prompt template
   - Can be built in ~20 lines
   - Handle conversation history
   - Production-ready architecture

7. **Practical Skills**
   - Created 413 vectors from 76 documents
   - Built working RAG system
   - Deployed with Gradio
   - Compared different encoders

### Best Practices

**Chunking**:
- Start with 1000 chars, 200 overlap
- Experiment with your data
- Consider document structure
- Test retrieval quality

**Encoders**:
- Develop with free models
- Upgrade for production
- Test with your specific data
- Visualize to understand behavior

**Retrieval**:
- Retrieve 3-10 chunks (k parameter)
- Consider conversation history
- Handle edge cases
- Measure quality

**Production**:
- Modular architecture
- Separate ingestion from querying
- Make implementations swappable
- Monitor and iterate

### Common Pitfalls to Avoid

1. **Confusing encoder with database**
   - Encoder creates vectors (affects quality)
   - Database stores vectors (affects speed)

2. **Ignoring conversation history**
   - LLMs are stateless
   - Must pass full history
   - Consider history in retrieval

3. **Not experimenting**
   - Chunking is empirical
   - Encoders vary by use case
   - Test and measure!

4. **Over-engineering**
   - Start simple
   - Add complexity only when needed
   - Measure impact of changes

## Next Steps

### Practice Exercises

1. **Try Different Chunk Sizes**:
   - Test 500, 1000, 2000 characters
   - Measure retrieval quality
   - Find optimal for your data

2. **Compare Encoders**:
   - Test all three encoders
   - Visualize differences
   - Measure retrieval accuracy

3. **Build Custom RAG**:
   - Use your own documents
   - Customize chunking strategy
   - Deploy with Gradio

4. **Advanced Features**:
   - Add source citations
   - Implement confidence scores
   - Handle multi-turn conversations

### Coming Up

**Advanced RAG Techniques**:
- Hybrid search (keyword + semantic)
- Re-ranking strategies
- Query expansion
- Evaluation metrics

**Production Optimization**:
- Caching strategies
- Batch processing
- Performance tuning
- Cost optimization

### Resources

**LangChain**:
- Documentation: https://python.langchain.com/
- GitHub: https://github.com/langchain-ai/langchain
- Community: https://github.com/langchain-ai/langchain/discussions

**Vector Databases**:
- Chroma: https://www.trychroma.com/
- FAISS: https://github.com/facebookresearch/faiss
- Pinecone: https://www.pinecone.io/
- Weaviate: https://weaviate.io/

**Encoders**:
- Sentence Transformers: https://www.sbert.net/
- OpenAI Embeddings: https://platform.openai.com/docs/guides/embeddings
- Hugging Face: https://huggingface.co/models?pipeline_tag=sentence-similarity

**Visualization**:
- t-SNE: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
- Plotly: https://plotly.com/python/

## Conclusion

Today you built a complete, working RAG system! You:
- Loaded 76 documents
- Created 413 chunks
- Generated 413 vectors
- Stored them in Chroma
- Visualized in 2D and 3D
- Built a RAG pipeline
- Deployed with Gradio

**This is production-ready code!** You can now:
- Add your own documents
- Customize the chunking
- Choose the best encoder
- Deploy for real users

RAG is one of the most important techniques in modern LLM applications. You now have hands-on experience building it from scratch.

**Congratulations!** You're now a RAG engineer! 🚀

## Appendix: Code Reference

### Complete RAG Implementation

```python
# ingest.py
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def fetch_documents():
    loader = DirectoryLoader("knowledge_base/", glob="**/*.md")
    return loader.load()

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documents)

def create_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./vector_db"
    )
    return vector_store

if __name__ == "__main__":
    docs = fetch_documents()
    chunks = create_chunks(docs)
    vector_store = create_embeddings(chunks)
    print(f"✅ Created {vector_store._collection.count()} vectors")
```

```python
# answer.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./vector_db",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-4o-mini")

def fetch_context(question):
    return retriever.invoke(question)

def answer_question(question, history=[]):
    context_docs = fetch_context(question)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    system_prompt = f"""Context: {context}
    
Answer based on the context."""
    
    messages = [SystemMessage(content=system_prompt)]
    for user_msg, ai_msg in history:
        messages.append(HumanMessage(content=user_msg))
        messages.append(AIMessage(content=ai_msg))
    messages.append(HumanMessage(content=question))
    
    response = llm.invoke(messages)
    return response.content
```

```python
# app.py
import gradio as gr
from implementation.answer import answer_question

demo = gr.ChatInterface(
    fn=answer_question,
    title="RAG Assistant",
    description="Ask questions about the knowledge base"
)

demo.launch()
```

This is your complete, production-ready RAG system!

