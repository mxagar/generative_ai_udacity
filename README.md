# Udacity Generative AI Nanodegree: Personal Notes

These are my personal notes taken while following the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

The Nanodegree asssumes basic data analysis skills with data science python libraries and databases, and has 4 modules that build up on those skills; each module has its corresponding folder in this repository with its guide Markdown file:

1. Generative AI Fundamentals: [`01_Fundamentals_GenAI`](./01_Fundamentals_GenAI/README.md).
    - Foundation Models
    - Fine-Tuning
2. Large Language Models (LLMs) & Text Generation: [`02_LLMs`](./02_LLMs/README.md).
    - Transformers and LLMs
    - Retrieval Augmented Generation (RAG) Chatbots
3. Computer Vision and Generative AI: [`03_ComputerVision`](./03_ComputerVision/README.md).
    - Generative Adversarial Networks (GANs)
    - Vision Transformers
    - Diffusion Models
4. Building Generative AI Solutions: [`04_BuildingSolutions`](./04_BuildingSolutions/README.md).
    - Vector Databases
    - LangChain and Agents

Additionally, it is necessary to submit and pass some projects to get the certification:

- Project 1: Apply Lightweight Fine-Tuning to a Foundation Model: [mxagar/llm_peft_fine_tuning_example](https://github.com/mxagar/llm_peft_fine_tuning_example).
- Project 2: Build Your Own Custom Chatbot - TBD.
- Project 3: AI Photo Editing with Inpainting - TBD.
- Project 4: Personalized Real Estate Agent - TBD.

Finally, also check some of my personal guides on related tools:

- My personal notes on the O'Reilly book [Generative Deep Learning, 2nd Edition, by David Foster](https://github.com/mxagar/generative_ai_book)
- My personal notes on the O'Reilly book [Natural Language Processing with Transformers, by Lewis Tunstall, Leandro von Werra and Thomas Wolf (O'Reilly)](https://github.com/mxagar/nlp_with_transformers_nbs)
- [HuggingFace Guide: `mxagar/tool_guides/hugging_face`](https://github.com/mxagar/tool_guides/tree/master/hugging_face)
- [LangChain Guide: `mxagar/tool_guides/langchain`](https://github.com/mxagar/tool_guides/tree/master/langchain)
- [LLM Tools: `mxagar/tool_guides/llms`](https://github.com/mxagar/tool_guides/tree/master/llms)
- [NLP Guide: `mxagar/nlp_guide`](https://github.com/mxagar/nlp_guide)
- [Deep Learning Methods for CV and NLP: `mxagar/computer_vision_udacity/CVND_Advanced_CV_and_DL.md`](https://github.com/mxagar/computer_vision_udacity/blob/main/03_Advanced_CV_and_DL/CVND_Advanced_CV_and_DL.md)
- [Deep Learning Methods for NLP: `mxagar/deep_learning_udacity/DLND_RNNs.md`](https://github.com/mxagar/deep_learning_udacity/blob/main/04_RNN/DLND_RNNs.md)

<!--
Finally, check these additional related courses:
- [Udacity Course on Small Datasets and Synthetic Data](https://www.udacity.com/course/small-data--cd12528)
-->

## Setup

A regular python environment with the usual data science packages should suffice (i.e., scikit-learn, pandas, matplotlib, etc.); any special/additional packages and their installation commands are introduced in the guides. A recipe to set up a [conda](https://docs.conda.io/en/latest/) environment with my current packages is the following:

```bash
# Create the necessary Python environment
# NOTE: specific folders might require their own environment
# and have their own requirements.txt
conda env create -f conda.yaml
conda activate genai

# Dependencies
pip-compile requirements.in
pip-sync requirements.txt

# If we need a new dependency,
# add it to requirements.in 
# (WATCH OUT: try to follow alphabetical order)
# And then:
pip-compile requirements.in
pip-sync requirements.txt

# When the repository is cloned, initialize and update the submodules 
git clone https://github.com/mxagar/generative_ai_udacity
git submodule update --init --recursive
```

## Credits

Many of the contents in this repository were created following the [Udacity Generative AI Nanodegree](https://www.udacity.com/course/generative-ai--nd608).

Mikel Sagardia, 2024.  
No guarantees.
