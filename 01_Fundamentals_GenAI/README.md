# Udacity Generative AI Nanodegree: Generative AI Fundamentals

These are my personal notes taken while following the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

The Nanodegree has 4 modules:

1. Generative AI Fundamentals.
2. Large Language Models (LLMs) & Text Generation.
3. Computer Vision and Generative AI.
4. Building Generative AI Solutions.

This folder & guide refer to the **first module**: Generative AI Fundamentals.

Mikel Sagardia, 2024.
No guarantees.

Overview of Contents:

- [Udacity Generative AI Nanodegree: Generative AI Fundamentals](#udacity-generative-ai-nanodegree-generative-ai-fundamentals)
  - [1. Introduction to Generative AI](#1-introduction-to-generative-ai)
    - [What Is Generative AI?](#what-is-generative-ai)
    - [Applications of Generative AI](#applications-of-generative-ai)
      - [Note on LLMs](#note-on-llms)
      - [Links of Examples](#links-of-examples)
    - [AI and Machine Learning Timeline](#ai-and-machine-learning-timeline)
    - [Training Generative AI Models](#training-generative-ai-models)
  - [2. Deep Learning Fundamentals](#2-deep-learning-fundamentals)
  - [3. Adapting Foundation Models](#3-adapting-foundation-models)
  - [4. Project: Applying Lightweight Fine-Tuning to a Foundation Model](#4-project-applying-lightweight-fine-tuning-to-a-foundation-model)


## 1. Introduction to Generative AI

Lesson objectives:

> - Identify industry applications, trends, and opportunities of Generative AI
> - Contextualize Generative AI within the broader history and landscape of machine learning and artificial intelligence
> - Describe the general process that popular Generative AI models use to generate outputs

Instructor: [Brian Cruz](https://www.linkedin.com/in/briancruzsf/).

### What Is Generative AI?

Examples of Generative AI:

- Text generation; e.g., ChatGPT
- Image generation; e.g., DALL-E
- Code generation; e.g., Github Copilot
- Audio generation: music and speech; e.g., [Meta's AudioCraft](https://ai.meta.com/resources/models-and-libraries/audiocraft/)

### Applications of Generative AI

In general, Generative AI has accelerate the ease to produce some content that previously required much more time. That implies people have become more productive; however, we should use it responsible to avoid destroying jobs, among other risks.

- Creative content generation
  - Artowrk synthesis: visual art pieces
  - Music composition: original musical pieces
  - Literary creation: written content
- Product development
  - Design optimization: refine designs
  - Rapid prototyping: concepts, visualization
  - Material exploration: predict and explore new materials
- Scientific research
  - Experiment simulation: physical testing less required
  - Data analysis and prediction
  - Molecular discovery: drug discovery
- Data augmentation
  - Image enhancement: new image varations
  - Text augmentation: diverse new texts
  - Synthetic data creation: new datasets from scratch
- Personalization
  - Content recommendation based on preferences and behavior
  - Bespoke product creation: tailored to individual specs
  - Experience customization: suit individual user preferences

#### Note on LLMs

> LLMs are able to create sentences that sound like they are written by humans, but they can struggle with questions that involve basic logic.
> This is because LLMs are primarily trained to be able to fill in missing words in sentences from the large corpora of text they are trained on.

Also, LLMs often avoid saying a simple *I don't know*, instead they try to hallucinate a made up answer. That is so because the principle they work on is precisely the hallucination of predicting the next word given the previous context.

#### Links of Examples

- [DeepMind: Millions of new materials discovered with deep learning](https://deepmind.google/discover/blog/millions-of-new-materials-discovered-with-deep-learning/)
- [Audi: Reinventing the wheel? “FelGAN” inspires new rim designs with AI](https://www.audi-mediacenter.com/en/press-releases/reinventing-the-wheel-felgan-inspires-new-rim-designs-with-ai-15097)
- [Paper: May the force of text data analysis be with you: Unleashing the power of generative AI for social psychology research](https://www.sciencedirect.com/science/article/pii/S2949882123000063)
- [Udacity Course on Small Datasets and Synthetic Data](https://www.udacity.com/course/small-data--cd12528)


### AI and Machine Learning Timeline

[Video: AI And Machine Learning Timeline](https://www.youtube.com/watch?v=W_n7kXdaC1Q)

![AI Timeline](./assets/ai_timeline.jpg)

### Training Generative AI Models

[Video: How Generative AI Models Are Trained](https://www.youtube.com/watch?v=cJ0VbfrN0iA)

Generative AI models are trained to learn an internal representation of a vast dataset. Then, after training, they can sample in the learned distribution to generate new but convincing data (images, text, etc.).

There are many ways to train generative AI models; we focus on two:

- LLMs: given a sequence of words (context) predict the next one; we reward the correct word and penalize the rest.
- Image generation models (e.g., diffusion models): they use the techniques in Variational Autoencoders; images are encoded into a latent space and then decoded back to reconstructued images. Bad reconstructions are penalized, good ones rewarded. Then, we use only the decoder part to generate new images feeding a latent vector.



## 2. Deep Learning Fundamentals

TBD.

:construction:

## 3. Adapting Foundation Models

TBD.

:construction:

## 4. Project: Applying Lightweight Fine-Tuning to a Foundation Model

TBD.

:construction:
