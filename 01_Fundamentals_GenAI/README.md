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
    - [Generation Algorithms](#generation-algorithms)
    - [Exercise: Generate Text using HuggingFace + GPT2](#exercise-generate-text-using-huggingface--gpt2)
    - [Other Generative AI Architectures](#other-generative-ai-architectures)
  - [2. Deep Learning Fundamentals](#2-deep-learning-fundamentals)
    - [Machine Learning and Deep Learning](#machine-learning-and-deep-learning)
    - [Notebooks: Machine Learning](#notebooks-machine-learning)
    - [Hugging Face](#hugging-face)
      - [Example: Sentiment Analysis, IMDB Dataset](#example-sentiment-analysis-imdb-dataset)
      - [Example: Trainer](#example-trainer)
    - [Notebooks: Pytorch + Hugging Face](#notebooks-pytorch--hugging-face)
    - [Transfer Learning](#transfer-learning)
    - [Notebook: Transfer Learning with MobileNetV3](#notebook-transfer-learning-with-mobilenetv3)
  - [3. Foundation Models](#3-foundation-models)
    - [Notebook: Foundation Model as Email Spam Classifier](#notebook-foundation-model-as-email-spam-classifier)
    - [GLUE and SuperGLUE Benchmarks](#glue-and-superglue-benchmarks)
      - [GLUE](#glue)
      - [SuperGLUE](#superglue)
    - [Training Data](#training-data)
      - [Biases](#biases)
      - [Links to Some Data Sources](#links-to-some-data-sources)
    - [Risks and the Bad Side of LLMs](#risks-and-the-bad-side-of-llms)
  - [4. Adapting Foundation Models](#4-adapting-foundation-models)
    - [RAG = Retrieval Augmented Generation](#rag--retrieval-augmented-generation)
    - [Prompt Design Techniques](#prompt-design-techniques)
      - [Prompt Tuning](#prompt-tuning)
      - [Exercises, Examples: Improving Prompts](#exercises-examples-improving-prompts)
    - [Using Probing to Train a Classifier](#using-probing-to-train-a-classifier)
  - [5. Project: Applying Lightweight Fine-Tuning to a Foundation Model](#5-project-applying-lightweight-fine-tuning-to-a-foundation-model)


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

In general, Generative AI has accelerated the ease to produce some content that previously required much more time. That implies people have become more productive; however, we should use it responsibly to avoid destroying jobs, among other risks.

- Creative content generation
  - Artwork synthesis: visual art pieces
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

### Generation Algorithms

Some generation algorithms:

- **Autoregression** for text generation: predict next word in a sequence, based on an initial seed and the previously detected words.
  - For each sequence, the probability of the words in a vocabulary to be the next word is predicted.
  - The models are trained to predict the next word or a word in between; i.e., we mask or remove some part of information and the model needs to find the original piece of information.
- **Latent Space Decoding**: we have trained an encoder and a decoder. We use the decoder to input random or manual vectors (latent vectors) and the decoder generates an expanded human-understandable representation.
- **Diffusion** models, often for images: they remove noise from a noisy (random) map. They iteratively remove noise to create a noise-free image.
  - The training consists in learning how to remove small steps of noise.
  - Again, we introduce small pieces of noise and try to find out the original piece of information without noise.

Common theme: we often take some information and add noise or mask it, and force the model to learn the obscured piece of information.
Then, the resulting models are able to generate new in-distribution information.

### Exercise: Generate Text using HuggingFace + GPT2

Notebook: [`lab/Exercise2-generating-one-token-at-a-time.ipynb`](./lab/Exercise2-generating-one-token-at-a-time.ipynb).

* Loads a pretrained causal language model and tokenizer using Hugging Face Transformers.
* Demonstrates how input text is tokenized into subword tokens.
* Computes next-token probabilities manually and appends the most likely token step-by-step.
* Allows interactive token-by-token generation to observe how text evolves.
* Compares manual generation with the model's built-in `.generate()` method for multi-token output.

### Other Generative AI Architectures

- **Generative Adversarial Networks (GANs)**
  - Generator + Discriminator.
  - Generator is trained to create new data samples; it takes random noise as input.
  - Discriminator is trained to discriminate whether the input is a real or generated sample.
  - They are able to generate very realistic images.
- **Recurrent Neural Networks (RNNs)**:
  - They predict the next element for an input sequence.
  - With each input, the RNN updates a hidden inner state, which is a combination of long and short-term memory.
- **Transformers**:
  - Text generation, translation.
  - They learn long-range dependencies in sequential data, and can generate new data, too.
  - Main difference & benefit of Transformers wrt. RNNs: they can work with the entire sequence in parallel! Thus, their training is much faster and scalable.

## 2. Deep Learning Fundamentals

See these resources of mine for deeper explanations:

- [mxagar/deep_learning_udacity](https://github.com/mxagar/deep_learning_udacity)
- [mxagar/machine_learning_coursera](https://github.com/mxagar/machine_learning_coursera)
- [mxagar/machine_learning_ibm](https://github.com/mxagar/machine_learning_ibm)
- [mxagar/tool_guides/hugging_face](https://github.com/mxagar/tool_guides/tree/master/hugging_face)

### Machine Learning and Deep Learning

**Machine Learning** concepts introduced in the videos of the GenAI course:

- Binary classifier
- Perceptron
- Weights and biases
- Activation function: sigmoid, ReLU
- Multi-Layer Perceptron
- Input, output and hidden layers
- Labeling a dataset
- Cost function
- Gradient descend
- Backpropagation
- Learning rate

**Pytorch** concepts introduced in the videos:

- Tensors
  - Multidimensional arrays: vectors, matrices, etc.
  - Several types
  - Linear algebra operations can be performed
- Neural nets as derived classes: `nn.Module`
- Loss functions: error computation between target/expected and model output
  - Classification: cross-entropy loss
  - Regression: Mean-Squared Error (MSE)
- Optimizers: adjust model parameters to minimize the cost/error
  - Gradients
  - Stochastic Gradient Descend
  - Learning rate
  - Momentum: add past weights
  - Adam: very good results, without much hyperparameter tuning
- Datasets and Data Loaders
  - Dataset class: represents and enables access to data in disk
  - Dataset loader class: loads samples from Dataset, e.g., in batches, shuffled, in parallel, etc.
- Training loop
  - Epochs, batches
  - 

Simple code examples:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 64)
        self.output_layer = nn.Linear(64, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.hidden_layer(x))
        return self.output_layer(x)

# Loss functions
ce_loss_function = nn.CrossEntropyLoss()
target_tensor = torch.tensor([1])
predicted_tensor = torch.tensor([[2.0, 6.0]])
loss_value = ce_loss_function(predicted_tensor, target_tensor)  # tensor(0.0181)

mse_loss_function = nn.MSELoss()
predicted_tensor = torch.tensor([320000.0])
actual_tensor = torch.tensor([300000.0])
loss_value = mse_loss_function(predicted_tensor, actual_tensor)
print(loss_value.item())  # 400000000.0

# Optimizers
sgd_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam_optimizer = optim.Adam(model.parameters(), lr=0.01)

# Create a toy dataset
class NumberProductDataset(Dataset):
    def __init__(self, data_range=(1, 10)):
        self.numbers = list(range(data_range[0], data_range[1]))

    def __getitem__(self, index):
        number1 = self.numbers[index]
        number2 = self.numbers[index] + 1
        return (number1, number2), number1 * number2

    def __len__(self):
        return len(self.numbers)

# Instantiate the dataset
dataset = NumberProductDataset(
    data_range=(0, 11)
)

# Access a data sample
data_sample = dataset[3]
print(data_sample)
# ((3, 4), 12)

# Instantiate the dataset
dataset = NumberProductDataset(data_range=(0, 5))

# Create a DataLoader instance
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Iterating over batches
for (num_pairs, products) in dataloader:
    print(num_pairs, products)
# [tensor([4, 3, 1]), tensor([5, 4, 2])] tensor([20, 12, 2])
# [tensor([2, 0]), tensor([3, 1])] tensor([6, 0])

# Training loop
for epoch in range(10):
    total_loss = 0.0
    for number_pairs, sums in dataloader:  # Iterate over the batches
        predictions = model(number_pairs)  # Compute the model output
        loss = loss_function(predictions, sums)  # Compute the loss
        loss.backward()  # Perform backpropagation
        optimizer.step()  # Update the parameters
        optimizer.zero_grad()  # Zero the gradients

        total_loss += loss.item()  # Add the loss for all batches

    # Print the loss for this epoch
    print("Epoch {}: Sum of Batch Losses = {:.5f}".format(epoch, total_loss))
```

### Notebooks: Machine Learning

Notebook: [`lab/Exercise1-classification-of-handwritten-digits-using-an-mlp.ipynb`](./lab/Exercise1-classification-of-handwritten-digits-using-an-mlp.ipynb)

* Loads the MNIST digit dataset using `sklearn.datasets.fetch_openml`.
* Trains a Multi-Layer Perceptron (MLP) classifier using `sklearn.neural_network.MLPClassifier`.
* Evaluates the model on both training and test datasets, reporting accuracy.
* Visualizes predictions on a sample of test images to manually inspect results.

### Hugging Face

**Hugging Face** concepts introduced in the videos of the GenAI course:

- Tokenizers
  - BERT
  - cased/uncased
  - Vocabulary
  - Subword tokenization
- Models
- Datasets
- Trainers
  - Truncating: shortening longer pieces of text to fit a certain size limit.
  - Padding: Adding filler data to shorter texts to reach a uniform length.
  - Batches: small, evenly divided parts of data.

#### Example: Sentiment Analysis, IMDB Dataset

```python
from IPython.display import HTML, display
from datasets import load_dataset
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the IMDB dataset, which contains movie reviews
# and sentiment labels (positive or negative)
dataset = load_dataset("imdb")

# Fetch a review from the training set
review_number = 42
sample_review = dataset["train"][review_number]

display(HTML(sample_review["text"][:450] + "..."))
# WARNING: This review contains SPOILERS. Do not read if you don't want some points revealed to you before you watch the
# film.
# 
# With a cast like this, you wonder whether or not the actors and actresses knew exactly what they were getting into. Did they
# see the script and say, `Hey, Close Encounters of the Third Kind was such a hit that this one can't fail.' Unfortunately, it does.
# Did they even think to check on the director's credentials...

if sample_review["label"] == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
# Sentiment: Negative

# Load a pre-trained sentiment analysis model
# IMPORTANT: the model was fine-tuned specifically for binary sentiment classification
# If we pass num_labels != 2 to it, it delivers random values, because the head needs to be re-trained
model_name = "textattack/bert-base-uncased-imdb"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize the input sequence
tokenizer = BertTokenizer.from_pretrained(model_name)
inputs = tokenizer("I love Generative AI", return_tensors="pt")

# Make prediction
with torch.no_grad():
    outputs = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    predicted_class = torch.argmax(probabilities)

# Display sentiment result
if predicted_class == 1:
    print(f"Sentiment: Positive ({probabilities[0][1] * 100:.2f}%)")
else:
    print(f"Sentiment: Negative ({probabilities[0][0] * 100:.2f}%)")
# Sentiment: Positive (88.68%)
```

#### Example: Trainer

```python
from transformers import (DistilBertForSequenceClassification,
    DistilBertTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset

model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

dataset = load_dataset("imdb")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    per_device_train_batch_size=64,
    output_dir="./results",
    learning_rate=2e-5,
    num_train_epochs=3,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()
```

### Notebooks: Pytorch + Hugging Face

Notebook: [`lab/Exercise2-pytorch-and-hugging-face-scavenger-huntscavenger-hunt.ipynb`](./lab/Exercise2-pytorch-and-hugging-face-scavenger-huntscavenger-hunt.ipynb)

Here is a summary of the notebook **“Exercise: PyTorch and HuggingFace scavenger hunt!”** in 5 bullet points:

* Introduces **basic PyTorch concepts**, including tensor creation, neural network layers (`torch.nn`), loss functions, and optimizers.
* Walks through constructing a simple training loop using PyTorch, with practice exercises and solutions.
* Transitions to **Hugging Face**, guiding the user to load a pretrained sentiment analysis model.
* Demonstrates how to use Hugging Face's Transformers library to tokenize text, run inference, and interpret sentiment predictions.
* Shows how to load a dataset from the Hugging Face `datasets` library for further experimentation.

### Transfer Learning

**Transfer Learning** concepts introduced in the videos of the GenAI course:

- Trasnfer learning: we train a model in a large domain dataset and then we train the head of that model on a smaller task-specific dataset.
  - The new head has the number of outputs we require for our task.
- Example large datasets: Common Crawl, ImageNet, LibriSpeech

![Transfer Learning](./assets/transfer_learning.jpg)

### Notebook: Transfer Learning with MobileNetV3

Paper: [MobileNetV3](https://paperswithcode.com/method/mobilenetv3)

Notebook: [`lab/Exercise3-transfer-learning-using-mobilenetv3.ipynb`](./lab/Exercise3-transfer-learning-using-mobilenetv3.ipynb)

* Loads the **Fashion-MNIST** dataset and defines helper functions to map label indices to names.
* Visualizes example training images to verify dataset integrity before training.
* Loads a **pretrained MobileNetV3 model** and modifies its classifier head for the Fashion-MNIST classification task.
* Trains the model using PyTorch, with device support for GPU, MPS, or CPU.
* Evaluates the trained model and visualizes correct and incorrect predictions on the test set.

## 3. Foundation Models

Characteristics of **Foundation Models**:

- Trained on several tasks
- Large/huge datasets used
- They require large amounts of resources to be trained
- They can generalize to new, unseen data
- They can perform tasks they were not trained for
- They can be adapted to specific domains and tasks
- Example: (Chat-) GPT model family, Bard/Gemini model family, etc.

In contrast, **traditional models** are trained on smaller and task-specific datasets, and require less resources.

The Foundation Model architecture is based on the **Transformer**:

- They handle sequential data in parallel, all at once, in contrast to previous RNNs.
- They have the novel **self-attention** mechanism, which is important for language modeling, where the relationship between words in a sequence are important to understand the sentence/sequence.

Some sizes of foundation models:

- Llama (2023) was trained on 4.7 TB of data
- Llama (2023) parameters: 6.7B - 65.2B

### Notebook: Foundation Model as Email Spam Classifier

Notebook: [`lab/Exercise1-use-a-foundation-model-to-build-a-spam-email-classifier.ipynb`](./lab/Exercise1-use-a-foundation-model-to-build-a-spam-email-classifier.ipynb)

* Loads a spam dataset (e.g., SMS spam) from Hugging Face using the `datasets` library.
* Defines label mappings and helper functions to preprocess and format SMS messages.
* Builds a **prompt-based classifier using a large language model (LLM)** to identify spam vs. ham.
* Evaluates classifier accuracy by comparing model outputs to true labels from the dataset.
* Iteratively improves the prompt and reruns evaluation to check for better accuracy and analyze misclassifications. **The improvement comes by adding some examples in the query/prompt, which increases the performance!**

### GLUE and SuperGLUE Benchmarks

Key points:

- GLUE = General Language Understanding Evaluation
- It is a collection of tests/tasks
- SuperGLUE is the successor of GLUE: it's more advanced and it appeared when models started achieving human parity in GLUE

#### GLUE

*Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy & Samuel R. Bowman.*  
[**GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding** (April 2018).](https://arxiv.org/abs/1804.07461)


| Short Name | Full Name                              | Description                                                           | Example                                                                                                                                  |
| ---------- | -------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| **CoLA**   | Corpus of Linguistic Acceptability     | Determine if a sentence is grammatically acceptable.                  | ✅ *"The boy is playing outside."*<br>❌ *"The boy playing is outside."*                                                                   |
| **SST-2**  | Stanford Sentiment Treebank            | Predict sentiment of a sentence (positive/negative).                  | *"This movie was fantastic!"* → **Positive**<br>*"The plot was boring and predictable."* → **Negative**                                  |
| **MRPC**   | Microsoft Research Paraphrase Corpus   | Classify if two sentences are paraphrases.                            | *"He is a doctor."* / *"He works as a medical professional."* → **Paraphrase**                                                           |
| **STS-B**  | Semantic Textual Similarity Benchmark  | Score how semantically similar two sentences are (0–5).               | *"A man is playing guitar."* / *"A person plays a musical instrument."* → **4.8**                                                        |
| **QQP**    | Quora Question Pairs                   | Determine if two questions are semantically equivalent.               | *"How can I learn Python?"* / *"What's the best way to study Python?"* → **Duplicate**                                                   |
| **MNLI**   | Multi-Genre Natural Language Inference | Decide if hypothesis is *entailment*, *neutral*, or *contradiction*.  | Premise: *"A man is playing a piano."*<br>Hypothesis: *"A man is making music."* → **Entailment**                                        |
| **QNLI**   | Question Natural Language Inference    | Determine if a passage answers a given question.                      | Question: *"Where is the Eiffel Tower?"*<br>Sentence: *"The Eiffel Tower is located in Paris, France."* → **Entailment**                 |
| **RTE**    | Recognizing Textual Entailment         | Binary entailment task: does one sentence logically follow the other? | Sentence1: *"Dogs bark loudly."*<br>Sentence2: *"Dogs make noise."* → **Entailment**                                                     |
| **WNLI**   | Winograd Natural Language Inference    | Resolve pronoun references based on nuanced context.                  | *"The city council refused the demonstrators a permit because they feared violence."*<br>**Who feared violence?** → **The city council** |

#### SuperGLUE

*Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy & Samuel R. Bowman.*
[**SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems** (May 2019)](https://arxiv.org/abs/1905.00537)


| Short Name  | Full Name                                        | Description                                                                           | Example                                                                                                                                                              |
| ----------- | ------------------------------------------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BoolQ**   | Boolean Questions                                | Answer a *yes/no* question given a passage.                                           | Passage: *"The Great Wall of China is over 13,000 miles long."*<br>Q: *"Is the Great Wall in Japan?"* → **No**                                                       |
| **CB**      | CommitmentBank                                   | Classify entailment, contradiction, or neutral based on a premise and hypothesis.     | Premise: *"John might go to the party."*<br>Hypothesis: *"John will definitely go to the party."* → **Contradiction**                                                |
| **COPA**    | Choice of Plausible Alternatives                 | Choose the more plausible cause or effect.                                            | Premise: *"The man broke his toe."*<br>Q: *"What was the cause?"*<br>A1: *"He dropped a hammer on his foot."*<br>A2: *"He got a promotion."* → **A1**                |
| **MultiRC** | Multi-Sentence Reading Comprehension             | Answer multi-choice questions from a passage, possibly with multiple correct answers. | Passage: *"...Alice and Bob went hiking... Bob packed food, Alice brought water..."*<br>Q: *"Who brought supplies?"*<br>Answers: ✅ **Alice**, ✅ **Bob**              |
| **ReCoRD**  | Reading Comprehension with Commonsense Reasoning | Fill in a blank with the correct entity from the passage.                             | Passage: *"...Marie Curie won two Nobel Prizes for her work on radioactivity..."*<br>Q: *"\_\_\_ won two Nobel Prizes for work on radioactivity."* → **Marie Curie** |
| **RTE**     | Recognizing Textual Entailment                   | Decide whether one sentence entails the other.                                        | Sentence1: *"A man is playing a piano."*<br>Sentence2: *"A man is making music."* → **Entailment**                                                                   |
| **WiC**     | Words in Context                                 | Determine if a word has the same meaning in two different contexts.                   | Sentence1: *"She gave him a ring."*<br>Sentence2: *"The phone began to ring."*<br>Word: *"ring"* → **Different**                                                     |
| **WSC**     | Winograd Schema Challenge                        | Resolve pronoun references using commonsense reasoning.                               | *"Emma thanked Julie because she helped her with the project."*<br>Q: *"Who helped Emma?"* → **Julie**                                                               |
| **AX-b**    | Broad Coverage Diagnostic                        | Diagnostic test for evaluating linguistic capabilities.                               | No fixed format — includes tasks like coreference, negation, and quantifiers to probe model behavior.                                                                |
| **AX-g**    | Winogender Schema Diagnostics                    | Diagnostic set to assess gender bias in coreference resolution. Coreference: when two words refer to the same concept.                      | *"The doctor hired the nurse because she was experienced."*<br>Tests whether “she” is wrongly assumed to be the nurse due to gender bias.                            |

### Training Data

LLMs and Foundation Models need to be trained with high quality data to perform nicely; these include:

- Websites (CommonCrawl): any kind of topic, text style, etc.
- Scientific papers (Arxiv): technical language and complex concepts
- Encyclopedias: general knowledge
- Books and literature: rich vocabulary and complex sentence structures
- Conversational posts: dialogues from TV scripts, colloquial speech
- Social media posts: modern jargon
- Legal documents: complex texts
- Multilingual texts: to allow for several languages

Texts are not used as they are; instead, they need to be preprocessed: cleaned, anonymized, filtered (for biases), formatted.

Some reference data points:

- 1GB text data = 1000 Books
- Llama v1 was trained with 4.7TB text data

#### Biases

**Biases** in the data can shape the model in an invisible yet impactful manner. These biases are often a reflection of historical data/events and can lead to the perpetuation of past errors.

Here are some biases:

| Bias Type                         | Description                                                                                | Example                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Selection Bias**                | Data used for training doesn’t represent the target population due to how it was selected. | A facial recognition model trained mostly on light-skinned individuals performs poorly on others.    |
| **Historical Bias**               | Biases present in society that are carried into the dataset and model.                     | A hiring algorithm trained on past resumes favors male applicants due to biased hiring history.      |
| **Confirmation Bias**             | Favoring data or patterns that align with preexisting beliefs or hypotheses.               | A sentiment classifier trained only on product reviews labeled by enthusiasts overstates positivity. |
| **Discriminatory Outcomes**       | Outputs that unfairly harm or disadvantage specific groups.                                | A loan approval model disproportionately denies applications from certain zip codes.                 |
| **Echo Chambers**                 | Feedback loops where biased outputs reinforce similar biased inputs.                       | A recommendation system keeps suggesting extreme political content based on initial clicks.          |
| **Measurement Bias**                     | Occurs when the tools or processes used to collect data systematically distort it. | Using outdated medical sensors that underreport symptoms in women compared to men.                |
| **Labeling Bias**                        | Introduced when human annotators apply inconsistent or subjective labels.          | Crowdworkers label images of “boss” as men more often than women, skewing downstream predictions. |
| **Simpson’s Paradox / Aggregation Bias** | Statistical trend visible in subgroups is hidden or reversed when combined.        | A model shows high accuracy overall but performs poorly on minority subgroups due to aggregation. |

Increasing organization diversity automatically decreases biases, because the environment becomes more tolerant to different views.

#### Links to Some Data Sources

- [https://commoncrawl.org/](https://commoncrawl.org/): Over 250 billion pages spanning 18 years, free. Greater than 1 PB. Unstructured and noisy.
- [https://www.githubarchive.org/](https://www.githubarchive.org/): Public repositories.
- [https://dumps.wikimedia.org/](https://dumps.wikimedia.org/): Available in many formats.
- [https://www.gutenberg.org/](https://www.gutenberg.org/): 75000 free eBooks.

### Risks and the Bad Side of LLMs

**Disinformation and misinformation** are both false or inaccurate information, but:

- Disinformation: intentional
- Misinformation: inadvertent

LLMs hallucinate and can help spread misinformation.

Additional risks:

- [AI has high data center energy costs](https://mitsloan.mit.edu/ideas-made-to-matter/ai-has-high-data-center-energy-costs-there-are-solutions)
- [Generative AI Has a Massive E-Waste Problem](https://spectrum.ieee.org/e-waste)
- [Exploring privacy issues in the age of AI ](https://www.ibm.com/think/insights/ai-privacy)
- **Over-reliance**

## 4. Adapting Foundation Models

Adaptation consists in customizing a pretrained Foundation Model to domain-specific tasks.

![Adaptation](./assets/adaptation.jpg)

Examples:

- Chatbot adaptation to banking
- LLM to structure medical records
- Specific instruction-based adaptation: translation of texts
- Fine-tuning of the model to be able to use private/corporate data

Relevant paper: [On the Opportunities and Risks of Foundation Models, Bommasani et al. (2021)](https://arxiv.org/abs/2108.07258)

* Coined and popularized the term **“foundation model.”**
* Proposed that foundation models represent a **paradigm shift** in AI, akin to general-purpose technologies.
* Laid groundwork for efforts like **model cards, data sheets**, and critical audits of large models.
* Opportunities:
  * Cross-task generalization
  * Rapid development of applications (e.g., via transfer learning)
  * Unifying architectures for vision, language, etc.
* Risks:
  * Bias, toxicity, and misinformation propagation
  * Environmental costs of training
  * Centralization of power (few labs controlling massive models)
  * Misuse (e.g., surveillance, manipulation)

### RAG = Retrieval Augmented Generation

We can fine-tune an LLM to adapt it to the domain or we can plug current information to its context using **Retrieval Augmented Generation (RAG)**.

For more information, see the co-located [mxagar/generative_ai_udacity/06_RAGs_DeepDive](https://github.com/mxagar/generative_ai_udacity/tree/main/06_RAGs_DeepDive).

![RAG Concept](./assets/rag_concept.jpg)

Concepts related to RAG:

- Context
- Semantic embeddings
- Cosine similarity
- Vector databases
- Keyword-based search in index tables
- Prompts

### Prompt Design Techniques

Key concepts:

- Prompt Tuning
  - The selection of words matters
  - The order in which we place the information matters
  - Usually, better putting the question at the end
- One/Few-shot prompting
  - We show a few examples (1-5) of what needs to be done
  - The model will try to copy out pattern
  - This is some kind of paradigm shift
- Zero-shot prompting: no examples provided, model requested directly
  - This is a kind of emergent property
  - We can use to classify texts, for instance
  - This is the most common way of interacting with LLMs, it seems normal, but it is actually incredible
- In-Context Learning: 
  - We provide information to be used in the context, e.g., 
    - task examples: mini labeled datasets
    - task descriptions: more abstract
  - Literature shows larger models are better in-context learners
- Chain-of-Thought
  - We display/show a series of logical steps to be followed
  - Example: we want to solve a logical problem, e.g.: "I bake 60 cookies, eat 10%, sell 20 from the rest, how many do I have?"
    - We provide a similar example in the prompt and ask a question to the example
    - We provide a step by step guide on how to compute the answer with the numbers
    - We finally add our question and ask the LLM
  - We can even add "think step by step"

Sometimes the boundaries between these concepts are not clear.

#### Prompt Tuning

A typical prompt template would be:

```python
f"""
{review}. In summary, the restaurant is
"""
```

But we can tune the prompt with several techniques, for instance **soft-prompting**. That consists in **learning** pre-pended tokens to optimize a task performance.

The result is a sequence of non human understandable tokens which is preprended to our prompt, which results in better performance!

![Soft Prompting](./assets/soft_prompting.jpg)

How soft-prompt header tokens can be obtained:

```python
# Define a small number n of learnable embeddings,
# e.g., 20 vectors of the same dimensionality as the model’s input embeddings
soft_prompt = torch.nn.Parameter(torch.randn(n, hidden_size))  # n x d

# For each input sequence (e.g., "Translate: Hello"),
# convert it to embeddings using the frozen model’s tokenizer and embedding layer.
# Concatenate the soft prompt embeddings in front of the input embeddings
input_embeds = model.embeddings(input_ids)  # shape: [batch, seq_len, d]
prompted_input = torch.cat([soft_prompt.expand(batch_size, -1, -1), input_embeds], dim=1)

# Feed the concatenated embeddings into the model.
# We may need to adjust the attention mask to include the soft tokens.
output = model(inputs_embeds=prompted_input, attention_mask=modified_mask)

# Freeze model params, only soft_prompt is learnable
for param in model.parameters():
    param.requires_grad = False
optimizer = torch.optim.Adam([soft_prompt], lr=1e-3)

# Compute loss + backpropagate
loss.backward()
optimizer.step()
```

In contrast to soft prompts, we have **hard prompting**: human trial/error to improve prompts.

More on soft-prompting: [Hugging Face PEFT conceptual guide - Soft prompts](https://huggingface.co/docs/peft/en/conceptual_guides/prompting)

#### Exercises, Examples: Improving Prompts

[Improve Your Queries Using Prompt Design Techniques](https://www.youtube.com/watch?v=awulyLb7v74)

> The video discusses an exercise focused on improving queries using prompt design techniques with a large language model. It begins by presenting a task where the model is asked to fill in missing answers based on specific rules for combining letters from a list of words. The initial attempts show that the model struggles with the task, sometimes providing incorrect answers.

> The presenter then explores different prompting techniques, including providing task descriptions, examples, and chain-of-thought prompts. They highlight that sometimes giving too much information can confuse the model, leading to worse performance. The video emphasizes the importance of experimenting with various prompt designs to find the most effective approach for specific tasks, illustrating how prompt design can significantly impact the model's output. 

### Using Probing to Train a Classifier




## 5. Project: Applying Lightweight Fine-Tuning to a Foundation Model

TBD.

:construction:
