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
  - [2. NLP Fundamentals](#2-nlp-fundamentals)
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

## 2. NLP Fundamentals

TBD.

:construction:

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
