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
    - Only 15% of the words are used for training, but the model performs well in downstream discriminative tasks: classification, etc.
  - GPT: Decoder-only
    - The next word is masked
    - Autoregressive: previous output is used  to enlarge the sequence and predict the next word/token
    - All the previous words/tokens are used, whereas the encoder training is bidirectional
    - These are the **generative** models in practice

![Transformer Model](./assets/transformer.jpg)

![Encoder Model](./assets/encoder_only.jpg)

![Decoder Model](./assets/decoder_only.jpg)

### Completion vs. Instruction Models


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
