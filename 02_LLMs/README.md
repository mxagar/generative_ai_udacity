# Udacity Generative AI Nanodegree: Large Language Models (LLMs) & Text Generation

These are my personal notes taken while following the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

The Nanodegree has 4 modules:

1. Generative AI Fundamentals.
2. Large Language Models (LLMs) & Text Generation.
3. Computer Vision and Generative AI.
4. Building Generative AI Solutions.

This folder & guide refer to the **second module**: Large Language Models (LLMs) & Text Generation.

Mikel Sagardia, 2024.
No guarantees.

Overview of Contents:

- [Udacity Generative AI Nanodegree: Large Language Models (LLMs) \& Text Generation](#udacity-generative-ai-nanodegree-large-language-models-llms--text-generation)
  - [1. Introduction LLMs](#1-introduction-llms)
    - [Encoder vs. Decoder Models](#encoder-vs-decoder-models)
    - [Completion vs. Instruction Models](#completion-vs-instruction-models)
    - [Data Flywheel](#data-flywheel)
    - [Fluency vs. Intelligence](#fluency-vs-intelligence)
    - [LLM Generation Parameters](#llm-generation-parameters)
    - [Demo Playground](#demo-playground)
    - [Prompts](#prompts)
    - [Demo: Prompts Using Chain-of-Thought](#demo-prompts-using-chain-of-thought)
    - [Open vs. Closed Models](#open-vs-closed-models)
    - [OpenAI Keys](#openai-keys)
    - [Exercise: Chatbot Memory](#exercise-chatbot-memory)
    - [LLM Limitations](#llm-limitations)
  - [2. NLP Fundamentals](#2-nlp-fundamentals)
    - [Introduction](#introduction)
    - [Encoding Text: Tokenization and Embeddings](#encoding-text-tokenization-and-embeddings)
  - [3. Transformers and Attention Mechanism](#3-transformers-and-attention-mechanism)
  - [4. Retrieval Augmented Generation](#4-retrieval-augmented-generation)
  - [5. Build Custom Datasets for LLMs](#5-build-custom-datasets-for-llms)
  - [6. Project: Build Your Own Custom Chatbot](#6-project-build-your-own-custom-chatbot)
    - [Notebooks](#notebooks)
    - [Project Requirements](#project-requirements)
    - [Interesting Links](#interesting-links)

## 1. Introduction LLMs

I know most of the contents, so I just write down the key concepts explained in the videos.

Historical recap of LLMs:

- 2017: Transformer model, from Google
- 2018: GPT, OpenAI
- BERT, RoBERTa, T5, ...
- Models continued to grow exponentially in size

See my additional notes in [mxagar/nlp_with_transformers_nbs](https://github.com/mxagar/nlp_with_transformers_nbs).

### Encoder vs. Decoder Models

- The original transformer was a Encoder-Decoder architecture; example model: T5
- However, wa can pull apart the Encoder and the Decoder and work with them independently
  - BERT: Encoder-only
    - Masked words need to be predicted during training
    - A classification head is added to the encoder output
    - The training is bidirectional because we have one/several masked words in a sequence and we need to attend both the previous and posterior words/tokens for each masked token.
    - The training process is called *masked language modeling*.
    - Only 15% of the words are used for training, but the model performs well in downstream discriminative tasks: classification, etc.
  - GPT: Decoder-only
    - The next word is masked
    - Autoregressive: previous output is used  to enlarge the sequence and predict the next word/token
    - All the previous words/tokens are used, whereas the encoder training is bidirectional
    - These are the **generative** models in practice

![Transformer Model](./assets/transformer.jpg)

![Encoder Model](./assets/encoder_only.jpg)

![Decoder Model](./assets/decoder_only.jpg)

Links:

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **[The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)**
- **[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)**
- [BERT](https://arxiv.org/abs/1810.04805)
- [GPT](https://openai.com/index/language-unsupervised/)

### Completion vs. Instruction Models

Basic Generative models perform *text completion*: they predict successively the next word/token and the text is completed.
Example use cases:

- Finish emails
- Fill in forms
- etc.

We can adapt those generative models to *instruction following*, so that we instruct them to perform a task and they carry it out (e.g., ChatGPT).
Example use cases:

- Translate
- Extract information
- Summarize
- etc.

Links:

- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [Instruction Tuning for Large Language Models: A Survey](https://arxiv.org/abs/2308.10792)

### Data Flywheel

Fine-tuning (for *text-completion*) and instruction fine-tuning (for *instruction following*) require new datasets; these don't need to be as big as the dataset used for pre-training the foundation model, but we need them, nevertheless.

Many options are possible, if the licenses allow them:

- Collect user interaction data; e.g., interaction with the chatbot/LLM. This is a positive feedback loop, a Flywheel.
- Generate a synthetic instruction-fine-tuning dataset with another model, e.g., by defining our desired instructions and asking an LLM to follow/complete them.
  `{"prompt": <existing_instruction>, "completion": <llm_generated_response>}`
- [Self-Alignment with Instruction Backtranslation](https://arxiv.org/abs/2308.06259): the inverse of the previous, i.e., we humans pick answers and ask an LLM to generate suited instructions.
  `{"prompt": <llm_generated_instruction>, "completion": <existing_document_chunk>}`
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)
- Generation of preference data from existing prompt/response LLM pairs. This can be used in Reinforcement Learning from Human Feedback (RLHF), but the feeback is still from an LLM. See: [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073).

### Fluency vs. Intelligence

LLMs are fluent, but not intelligent. Although this is quite controversial.

Language fluency is a human capability, and LLMs seem to have super-human language capabilities.

We could say that language is a step function over all species, and humans have extensively exploited it.

### LLM Generation Parameters

- Model
- Temperature: token-probability pairs are flattened with higher temperatures, so more tokens are likely, we get more creative. Take into account that we **sample tokens/words** in that probability distribution to pick the next token/word.
  - Alternative: greedy decoding, i.e., we pick the token/word with the highest probability.
- Context length or attention window, composed of:
  - System prompt
  - Chat history: so that the LLM knows/has memory of the conversation
  - User prompt
  - Maximum length, `max_tokens`: pre-allocated for the LLM response
  - Note: each new generated token needs to be able to attend all previous tokens!
- Top `p` or top `k`: we chop off all tokens below a probability `p`, so we sample only above a given `p` probability. Similarly we sample on the top `k` tokens/words.
- Penalties (repetition or frequency): the `p` values of already used tokens/words are reduced to avoid repetition. Repetition is a common phenomenon in smaller models.

Example: `temperature = 0` and `top_k = 1` leads to greedy decoding.

Links:

- [HF Blog: How to generate text: using different decoding methods for language generation with Transformers](https://huggingface.co/blog/how-to-generate)
- [OpenAI Reference: Create chat completion](https://platform.openai.com/docs/api-reference/chat/create)
- [OpenAI Platform](https://platform.openai.com/docs/overview)
- [OpenAI Playground](https://platform.openai.com/playground/)
- My notes on the HF Book chapter about test generation: [mxagar/nlp_with_transformers_nbs/chapter-5-text-generation](https://github.com/mxagar/nlp_with_transformers_nbs?tab=readme-ov-file#chapter-5-text-generation)


### Demo Playground

After creating an account, we can use the [OpenAI Playground](https://platform.openai.com/playground/). Recommendations:

- Be as specific as possible
- Use few-shot prompts to improve responses
- Avoid vicious circles: the mode might fall into repeating a sequence or words and then tries to repeat them endlessly. Intuitively, that's understandable: we pass as input the generated response, and since the sequence is being repeated, the model keeps repeating it. More information: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751).

### Prompts

The model is asked to predict the probabilities of the tokens to be the next selected one (`y`), given a previous sequence (`x`, incl. the prompt) and the model weights (`theta`).

    p(x|y,theta)

Since model weights (`theta`) are constant, the variable we have to improve the response is `x`, i.e., **the prompt and the overall context**.

- Remember LLMs are very good in language fluency, conditioned on the prompt.
- The more precise the context, the less hallucinations.

The prompt is composed by several parts:

- System prompt: personality, role, main task
- Chat history: conversation so far
- Augmented user prompt: we enrich the user request with specific orders proven to increase efficiency
  - Trigger phrases like *Think step by step*
  - Other Chain-of-thought prompts: *Define step by step the reasoning followed to provide the answer*
  - Giving some examples
  - etc.

```python
[
  {"role": "system", "content": "You are an AI assistant..."},  # system prompt
  {"role": "user", "content": "Can you..."},  # chat history
  {"role": "assistant", "content": "Sure..."},  # chat history
  {"role": "user", "content": "Why did you..."},  # user prompt
  
]
```

### Demo: Prompts Using Chain-of-Thought

Create an account at [https://api.together.xyz/](https://api.together.xyz/).

Notebook: [`lab/demo_cot_and_triggers.ipynb`](./lab/demo_cot_and_triggers.ipynb)

In the notebook, we ask Llama 2-7B the number of tokens teh user prompt can contain given the context window size and the size of the appended prompt parts; LLMs famously fail in math exercises.

System prompts are constructed in two ways:

1. Answering the question first or
2. Providing the reasoning first

The latter works best, i.e.: provide a reasoning + finally provide an answer based on the reasoning.

Also, a simple trigger phrase is used: *Think step by step*; it does not perform better, but that's maybe not generalizable.

Links (mainly on Chain-of-Thought):

- [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916)
- [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
- [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805)

### Open vs. Closed Models

Closed models are like black boxes, accessed via API.
We can control the weights of open models, but often their performance is not as good as the one of the closed models.

### OpenAI Keys

We have 5 USD for OpenAI usage via [Vocareum](https://www.vocareum.com/).

Our budget can be checked in the `Cloud Resources` tab.

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(".env", override=True)
# Set a .env file with the credentials
# OPENAI_API_KEY=xxx
# OPENAI_BASE_URL=https://openai.vocareum.com/v1
# OPENAI_BASE_URL=https://api.openai.com/v1

client = OpenAI(
    base_url = os.getenv("OPENAI_BASE_URL"),
    api_key = os.getenv("OPENAI_API_KEY"),
)
```

### Exercise: Chatbot Memory

Notebook: [`lab/chatbot_memory_management.ipynb`](./lab/chatbot_memory_management.ipynb).

The new OpenAI completions interface was missing, so I updated it.

Key contents:

- **Implements a chatbot with memory**: The notebook builds a conversational agent that keeps track of conversation history across multiple turns.
- **Manages the attention window**: It demonstrates strategies to truncate conversation history dynamically to avoid exceeding the model's token limit.
- **Uses OpenAI GPT models via `OpenAI.chat.completions.create` API**
- **Provides truncation strategies**: Includes two truncation modes: 
  - Simple truncation (dropping oldest messages) 
  - and more selective truncation (removing the oldest user/assistant exchanges while preserving the system prompt).
- **Includes error handling and retry logic**: When prompts exceed the model's capacity, the code captures errors, adjusts the prompt, and retries until successful.

Insights:

- **Memory management is essential** when building chatbots that preserve context—without truncation, prompts will eventually exceed token limits.
- **Different truncation strategies** have different trade-offs: simpler approaches are easier to implement, while more sophisticated ones better preserve important context.
- **Clear separation of system prompts, user inputs, and model responses** improves maintainability and interpretability of conversation history.

Caveat: In the notebook, the conversation is flattened to a single string which contains the entire conversation history. That was necessary in the old API, but now it's possible to pass a list of messages, without any flattening:

```python
messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm fine, how can I help?"},
    {"role": "user", "content": "Tell me a joke."}
]
```

### LLM Limitations

Recall what LLMs do: They predict the next token/word probabilities (`y`) given the input sequence (`x`) and the constant model weights (`theta`):

    p(y|x, theta)

Therefore: The more specific and rich the input `x` the more the model will rely on it, otherwise the more will it rely on its weights.

Challenging tasks:

- Avoiding hallucinations
- Repeating/using difficult to tokenize texts, such as URLs 
- Maths problems

Approaches to overcome these limitations:

- Rich context
- Using tools: calculators, etc.

## 2. NLP Fundamentals

I knew already many concepts, so I just collect the concept names/topics.

For a deeper explanation of them, see:

- [mxagar/nlp_guide](https://github.com/mxagar/nlp_guide)
- [mxagar/nlp_with_transformers_nbs](https://github.com/mxagar/nlp_with_transformers_nbs)

### Introduction

Key ideas:

- NLP = Natural Language Processing
- Natural Language: ambiguity is a feature, in contrast to structured languages (e.g., programming languages).
- NLP applications
  - Speech recognition
  - Text classification
  - Machine language translation
  - Text summarization: extractive vs. abstractive summaries
  - QA: Question-Answering (similar to summarization), also extractive and/or abstractive
  - Chatbots, conversational agents
- Challenges in NLP
  - Relies on context
  - Nuanced: idioms, sarcasm
  - Ambiguity, references within the text
  - Misspelling
  - Biases
  - Labeling

### Encoding Text: Tokenization and Embeddings

Key ideas:

- Tokenization and embeddings are used to encode text
  - Tokens: unitary and discrete chunks
  - Embeddings: continuous vectors which encode meaning and context
- Tokenization steps
  - Normalization: clean, lowercase, remove punctualization, etc. We can decide the degree
  - Pretokenization: split by spaces, i.e., we words and symbols
  - Tokenization: split words and symbols into tokens (sub-word tokenization)
  - Postprocessing
- HuggingFace tokenizer: see usage in notebook and code summary below.
  - Encoding and decoding: words <-> tokens <-> ids
  - Maximum model length: 512 (tokens) for BERT
  - Special tokens: unknown, B/EOS, padding, classification, etc.
- Embeddings
  - Vectorization methods: Bag-of-words, one-hot encoding, TF-IDF
    - All of these lack of meaning capturing, context awareness
  - Embeddings are vectors which capture meaning: similar tokens have similar vectors
    - We can perform math operations with meanings!

![Tokenization and Embeddings](./assets/tokenization_embeddings.jpg)

![Embeddings Operations](./assets/embeddings_operations.jpeg)

Notebooks: 

- [`lab/hugging-face-tokenizer.ipynb`](./lab/hugging-face-tokenizer.ipynb)
- [`lab/hugging-face-tokenizer-properties.ipynb`](./lab/hugging-face-tokenizer-properties.ipynb)

**HuggingFace Tokenizer** code summary:

```python
# Choose & download a pretrained tokenizer to use
# BERT (encoder-only) and CASED: it cares about capitalization
my_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

# Simple method getting tokens from text
raw_text = '''Rory's shoes are magenta and so are Corey's but they aren't nearly as dark!'''
tokens = my_tokenizer.tokenize(raw_text)
print(tokens)
# Sub-word tokenization: words are split into sub-words
# In the case of the BERT tokenizer, the successive split parts are prefixed with '##'
# ['Rory', "'", 's', 'shoes', 'are', 'mage', '##nta', 'and', 'so', 'are', 'Corey', "'", 's', 'but', 'they', 'aren', "'", 't', 'nearly', 'as', 'dark', '!']

# This method also returns special tokens depending on the pretrained tokenizer
# Special tokens are used to mark the beginning and end of a sequence, etc.
# BERT uses [CLS] for the start of a sequence and [SEP] for the end
# [UNK] is used for unknown tokens or words/tokens out-of-vocabulary (e.g., often emojis)
detailed_tokens = my_tokenizer(raw_text).tokens()
print(detailed_tokens)
# ['[CLS]', 'Rory', "'", 's', 'shoes', 'are', 'mage', '##nta', 'and', ..., [SEP]']

# Way to get tokens as integer IDs
print(my_tokenizer.encode(raw_text))
# [101, 14845, 112, 188, 5743, 1132, 27595, 13130, 1105, 1177, 1132, 19521, 112, 188, 1133, 1152, 4597, 112, 189, 2212, 1112, 1843, 106, 102]

# Tokenizer method to get the IDs if we already have the tokens as strings
detailed_ids = my_tokenizer.convert_tokens_to_ids(detailed_tokens)
print(detailed_ids)
# [101, 14845, 112, 188, 5743, 1132, 27595, 13130, 1105, 1177, 1132, 19521, 112, 188, 1133, 1152, 4597, 112, 189, 2212, 1112, 1843, 106, 102]

# Returns an object that has a few different keys available
# It returns a dictionary/object
my_tokenizer(raw_text)
# {'input_ids': [101, 14845, 112, 188, ...], 'token_type_ids': [0, 0, 0, 0, ...], 'attention_mask': [1, 1, 1, 1, ...]}

# Typical call:
print(my_tokenizer(raw_text).input_ids)
# [101, 14845, 112, 188, ...]

# Integer IDs for tokens
ids = my_tokenizer.encode(raw_text)
# The inverse of the .encode() method: .decode()
my_tokenizer.decode(ids)
# "[CLS] Rory ' s shoes are magenta and so are Corey ' s but they aren ' t nearly as dark! [SEP]"

# To ignore special tokens (depending on pretrained tokenizer)
my_tokenizer.decode(ids, skip_special_tokens=True)
# "Rory ' s shoes are magenta and so are Corey ' s but they aren ' t nearly as dark!"

# List of tokens as strings instead of one long string
my_tokenizer.convert_ids_to_tokens(ids)
# ['[CLS]', 'Rory', "'", 's', 'shoes', ...

# tokenizer.model_max_length: the maximum amount of tokens that the tokenizer/model takes per input
my_tokenizer.model_max_length # 512

# tokenizer.all_special_tokens: the special tokens used by the tokenizer
my_tokenizer.all_special_tokens # ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']

# tokenizer.unk_token: the unknown token used by the tokenizer
# tokenizer.bos_token: the beginning of sequence token used by the tokenizer
# tokenizer.eos_token: the end of sequence token used by the tokenizer
# tokenizer.pad_token: the padding token used by the tokenizer
# tokenizer.cls_token: the classification token (aka. class of input) used by the tokenizer
my_tokenizer.unk_token # '[UNK]'
```

## 3. Transformers and Attention Mechanism

TBD.

:construction:

## 4. Retrieval Augmented Generation

TBD.

:construction:

## 5. Build Custom Datasets for LLMs

TBD.

:construction:

## 6. Project: Build Your Own Custom Chatbot

TBD.

:construction:

### Notebooks

- Project: [mxagar/rag-app-examples](https://github.com/mxagar/rag-app-examples)

### Project Requirements



### Interesting Links

- [mxagar/azure-rag-app](https://github.com/mxagar/azure-rag-app)
- [aws-samples/aws-genai-llm-chatbot](https://github.com/aws-samples/aws-genai-llm-chatbot)
- [Azure-Samples/azure-search-openai-demo](https://github.com/Azure-Samples/azure-search-openai-demo/)
