# RAG from Scratch: LangChain

This project includes resources from [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) as a submodule. The original author of that repository is Lance Martin, from Langchain.

- [RAG from Scratch by Langchain (Github)](https://github.com/langchain-ai/rag-from-scratch)
- [RAG from Scratch by Langchain (Youtube & Freecodecamp)](https://www.youtube.com/watch?v=sVcwVQRHIc8)

## Table of Contents

- [RAG from Scratch: LangChain](#rag-from-scratch-langchain)
  - [Table of Contents](#table-of-contents)
  - [Setup](#setup)
    - [LangSmith and OpenAI](#langsmith-and-openai)
    - [All Environment Variables](#all-environment-variables)
    - [Original LangChain Repository](#original-langchain-repository)
  - [Part 1: Introduction](#part-1-introduction)

## Setup

The only necessary libraries are these:

```bash
pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
```

However, I created a fresh new basic environment with `conda`:

```bash
# Create environment (Python 3.10, pip & pip-tools)
conda env create -f conda.yaml
# Activate environment
conda activate rag

# Generate pinned dependencies and install/sync
pip-compile requirements.in --verbose
pip-sync requirements.txt

# Install package as editable: changes are immediately reflected without reinstalling
# This requires a setup.py, as explained below
pip install -e .

# If we need a new dependency,
# add it to requirements.in 
# (WATCH OUT: try to follow alphabetical order)
# And then:
pip-compile requirements.in
pip-sync requirements.txt
```

### LangSmith and OpenAI

We should create a a free developer account at **LangSmith**; there are two possible endpoints:

- [LangSmith (EU)](https://eu.smith.langchain.com/)
- [LangSmith (US)](https://smith.langchain.com)

I created both accounts, but the EU one seems to have issues when some hub repositories are accessed, so I ended up using the US one.

Once we have the LangSmith US account, we set a new project (e.g., `rag-from-scratch`) and get the environment variables to access it.

Additionally, we need an [OpenAI Platform Account](https://platform.openai.com/) and a project API key from it.

### All Environment Variables

I have a `.env` file as the following:

```bash
# Obtained from LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"
LANGCHAIN_API_KEY="xxx"
LANGCHAIN_PROJECT="xxx"
# OpenAI Project API Key
OPENAI_API_KEY="xxx"
```

### Original LangChain Repository

The original LangChain repository is added as a submodule:

```bash
# Add and initialize the LanChain repo as a submodule
cd .../generative_ai_udacity
git submodule add https://github.com/langchain-ai/rag-from-scratch.git 06_RAGs_DeepDive/01_RAG_from_Scratch/notebooks/rag-from-scratch
git submodule init
git submodule update

# Add the automatically generated .gitmodules file to the repo
git add .gitmodules 06_RAGs_DeepDive/01_RAG_from_Scratch/notebooks/

# When my repository is cloned, initialize and update the submodule 
git clone https://github.com/mxagar/generative_ai_udacity
git submodule update --init --recursive
```

## Part 1: Introduction

Resources:

- Video: [RAG from Scratch: Part 1](https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=1)
- Notebooks: 
  - Original: [`rag_from_scratch_1_to_4.ipynb`](./notebooks/rag-from-scratch/rag_from_scratch_1_to_4.ipynb)
  - Mine: [`RAG_Scratch_Part_01.ipynb`](./notebooks/RAG_Scratch_Part_01.ipynb)

LLMs have not seen all the data we car about: recent data, private data.
Additionally, we have the context, to which we can inject data.
This is related to the LLM OS, coined by Karpathy, where LLMs are connected to external data.

![LLM OS Karpathy](./assets/llm_os_karpathy.png)

We have three basic RAG stages/components:

1. Indexing: documents are indexed.
2. Retrieval: Given a query, the relevant documents are obtained
3. Generation: An answer is formulated by the LLN given the query and the retrieved documents

However, there are more advanced RAG systems that go beyond those 3 components; they include:

- Query transformation
- Routing
- Query construction
- Indexing
- Retrieval
- Generation

![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)

This tutorial builds up from basics to advanced.

Very simple RAG example shown in []`RAG_Scratch_Part_01.ipynb`](./notebooks/RAG_Scratch_Part_01.ipynb)

- A basic RAG is built, where a blog is vectorized and used to build a chatbot.
- The `hub` does not work if the EU endpoint is used, i.e., pulling the template fails; thus, I ended up using the US API.
- LangSmith: This is some kind of tracking tool, with a nice UI: [LangSmith](https://smith.langchain.com); it's similar to MLflow, but built for LLMs and LangChain-related operations.

![LangSmith UI](./assets/langsmith_ui_1.png)

```python
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate

load_dotenv(override=True, dotenv_path="../.env")

#### INDEXING ####

# Load Documents
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

#### RETRIEVAL and GENERATION ####

# Prompt
prompt = hub.pull("rlm/rag-prompt") # This is not working anymore...
# Alternative:
#prompt = PromptTemplate(
#    input_variables=["context", "question"],
#    template="Given the following context: {context}, answer the question: {question}"
#)

# LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("What is Task Decomposition?")
```
