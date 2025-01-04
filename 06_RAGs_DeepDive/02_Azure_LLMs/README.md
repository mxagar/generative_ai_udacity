# Operationalizing LLMs on Azure

This folder includes my notes on the course [Operationalizing LLMs on Azure](https://www.coursera.org/learn/llmops-azure), which is the second module of the Coursera Specialization [Large Language Model Operations (LLMOps)](https://www.coursera.org/specializations/large-language-model-operations).

For a guide on Azure, check my notes in [mxagar/azure_guide](https://github.com/mxagar/azure_guide).

## Table of Contents

- [Operationalizing LLMs on Azure](#operationalizing-llms-on-azure)
  - [Table of Contents](#table-of-contents)
  - [0. Setup](#0-setup)
  - [1. Introduction to LLMOps with Azure](#1-introduction-to-llmops-with-azure)
    - [1.1 Introduction to Azure and Its AI Services](#11-introduction-to-azure-and-its-ai-services)
      - [Azure Machine Learning](#azure-machine-learning)
      - [Azure OpenAI](#azure-openai)
    - [1.2 Overview of LLMs](#12-overview-of-llms)
    - [1.3 LLM Deployment in Azure](#13-llm-deployment-in-azure)
      - [Azure Machine Learning](#azure-machine-learning-1)
      - [Azure AI Content Safety](#azure-ai-content-safety)
      - [Azure Open AI (Studio - or Azure AI Foundry)](#azure-open-ai-studio---or-azure-ai-foundry)
  - [2. LLMs with Azure](#2-llms-with-azure)
    - [2.1 Azure Machine Learning and LLMs](#21-azure-machine-learning-and-llms)
      - [Check Quotas](#check-quotas)
      - [Create Compute Instances](#create-compute-instances)
      - [Deploy a Model](#deploy-a-model)
    - [2.2 Azure OpenAI Service](#22-azure-openai-service)
    - [2.3 Azure OpenAI APIs](#23-azure-openai-apis)
  - [3. Extending with Functions and Plugins](#3-extending-with-functions-and-plugins)
    - [3.1 Improved Prompts with Semantic Kernel](#31-improved-prompts-with-semantic-kernel)
    - [3.2 Extending Results with Functions](#32-extending-results-with-functions)
    - [3.3 Using Functions with External APIs](#33-using-functions-with-external-apis)
  - [4. Building an End-to-End Application in Azure](#4-building-an-end-to-end-application-in-azure)
    - [4.1 Architecture](#41-architecture)
      - [Azure AI Search](#azure-ai-search)
      - [Github Actions](#github-actions)
      - [Azure AI Document Intelligence](#azure-ai-document-intelligence)
      - [Extra Exercises](#extra-exercises)
    - [4.2 RAG with Azure AI Search](#42-rag-with-azure-ai-search)
    - [4.3 Deployment and Scaling with Github Action](#43-deployment-and-scaling-with-github-action)

## 0. Setup

Steps:

- Create an Account in Azure: [mxagar/azure_guide/01_fundamentals#12-basic-demos](https://github.com/mxagar/azure_guide/tree/main/01_fundamentals#12-basic-demos); I used an account linked to my Github.
- Create a Python environment (see below).

Recipe to create a `conda` environment with provided dependency files:

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

## 1. Introduction to LLMOps with Azure

### 1.1 Introduction to Azure and Its AI Services

For an overview of Azure, check my guides: [mxagar/azure_guide/01_fundamentals](https://github.com/mxagar/azure_guide/tree/main/01_fundamentals)

An important portal for learning Azure: [https://learn.microsoft.com/](https://learn.microsoft.com/)

- Documentation: Generic topics explained in detail.
- Training
- [**Code Samples**](https://learn.microsoft.com/en-gb/samples/browse/): We have very specific and useful examples here!

It's important to distinguish between two important AI services:

- Azure OpenAI: These are OpenAI models
  - We can open **Azure OpenAI Studio** and check them.
  - If we use them, we get OpenAI model clones running on our Subscription only for us!
- Azure Machine Learning: Here, we can create instances, which are then opened in **Azure Machine Learning Studio**
  - In the Studio, we have access to all sorts of models, not the OpenAI models.
  - We have models for vision, language, etc.
  - We have Endpoints, etc.

#### Azure Machine Learning

An **Azure Machine Learning Workspace** is created in [Azure Machine Learning](https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-machine-learning?view=azureml-api-2). The *container registry* is left empty (i.e., *None*).

![Create Azure ML Workspace](./assets/create_azure_ml_workspace.png)

Several services are deployed (it takes some minutes) in the selected/created RG.

![Azure ML Resource Group Overview](./assets/azure_ml_rg.png)

Once created, we go to the `Azure Machine Learning workspace` service instance and click on `Launch Studio`.

![Azure ML Studio](./assets/azure_ml_studio.png)

There we see we have many options on the left menu:

- Model catalog (OpenAI models appear again, but we are redirected to OpenAI Studio).
- Notebooks (also samples).
- (Our) Models, in a registry.
- Endpoints.
- Compute instances: we can create compute instance with GPU and attach them to our Notebooks.

:warning: **IMPORTANT**: When we deploy an Azure Machine Learning Workspace **without** anything in it, it does not incur in any fixed costs; but when we start adding compute & Co., then we'll have costs.

#### Azure OpenAI

[Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview) is a wrapper for the OpenAI models on Azure, i.e., we can deploy and access custom OpenAI models via Azure using it. In addition, it offers other functionalities for user convenience: fine-tuning, network access, etc.

We create a resource instance on the Azure Platform:

![Create Azure OpenAI Service](./assets/create_azure_open_ai.png)

- Basics: see snapshot.
  - Check the [Azure OpenAI Pricing](https://azure.microsoft.com/en-gb/pricing/details/cognitive-services/openai-service/)
- Network: we select `all networks`, i.e., we can access the resource from the Internet. We could restrict that to selected networks or completely shut down network access.

:warning: **IMPORTANT**: When we deploy an Azure OpenAI instance **it incurs fixed costs even if we don't use it!**

Once created, we go to the resource:

- (Left) Resource Management > Keys and Endpoints: Endpoints + Credentials to access via Internet are here!
- (Left) Overview: general information
  - Here we should have also a link to the **Azure Open AI Studio** (old) or **Azure AI Foundry**.
  - In the Studio/Foundry, we can configure and deploy models, get credentials, try them in the playground, fine-tune them, etc.

![Azure AI Foundry](./assets/azure_ai_foundry.png)

In **Azure AI Foundry**:

- In the **Shared Resources** section we have management tools:
  - Deployments: they should appear here, with their configuration properties.
  - Quotas: each region-model-subscription has a limit in tokens/minute, etc.
  - Data files: for fine-tuning
  - Safety and Security: content filters.
  - Quotas.
  - etc.
- In the **Playground**, we can use/chat with the deployed models.
  - We can use our deployed model in the chat.
  - If we click on `View code` we see the Python code for interaction.
- Deploying an Azure OpenAI model here is much easier than deploying any other readily available model.
  - No need to define Compute and/or configure Endpoints.
- Also, once deployed, we can change its configuration, e.g., change its Tokens Per Minute (TPM) quota.
  - One important feature is using the dynamic quota, i.e., capacity increases dynamically if general subscription quota is available.
  - Note also that we have a content filter by default! The answer will contain some assessment wrt. our filter!

![Azure AI Foundry: Playground](./assets/auzure_ai_foundry_playground.png)

![Azure AI Foundry: Deployments](./assets/auzure_ai_foundry_deployments.png)

![Azure AI Foundry: OpenAI Model Update](assets/azure_openai_model_update.png)

In the notebook [`01_azure_open_ai_basics.ipynb`](./notebooks/01_azure_open_ai_basics.ipynb) I show how to use the OpenAI deployment programmatically via REST; I used the API key and the Endpoint obtained from the Azure AI Foundry (Deployments).

- Note that we have access to the Endpoint URL and the keys in two spots:
  - In the Azure Portal, if we select the Azure OpenAI resource: Resource management > Keys and Endpoint.
  - In the Azure AI Foundry: Select deployment: Details view.
- Also, note that the Endpoint URL can have several forms:
  - A specific form is suited for `curl` or `requests` calls, because it contains all the elements necessary to contact our model.
  - A generic is shorter, it's the base of the specific; it is used by libraries, like `openai`.

In the Chat Playground, if we click on `View code` we see the Python code for interaction, which is different to the simple REST API call. The example code from `View code` has more options; I also tested it in the notebook.

```python
### -- Simple API Call

import os
import requests
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path=".env")

# Define your Azure OpenAI details
AZURE_OPENAI_ENDPOINT_URI = os.getenv("AZURE_OPENAI_ENDPOINT_URI")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

# Set up the request headers
headers = {
    "Content-Type": "application/json",
    "api-key": AZURE_OPENAI_API_KEY,
}

# Define the request body
data = {
    "messages": [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Tell me a joke."},
    ],
    "max_tokens": 100,
}

# Send the POST request
response = requests.post(
    f"{AZURE_OPENAI_ENDPOINT_URI}",
    headers=headers,
    json=data,
)

# Handle the response
if response.status_code == 200:
    result = response.json()
    print("Response:")
    print(result["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code} - {response.text}")
# Response:
# Why don't scientists trust atoms? 
# 
# Because they make up everything.

### -- Code Example from Playground

import os  
import base64
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True, dotenv_path=".env")

# Now, the URL is shorter and we add the endpoint and deployment name
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",  
)

#Prepare the chat prompt 
chat_prompt = [
    {
        "role": "system",
        "content": "You are an AI assistant that helps people find information."
    }
] 
    
# Include speech result if speech is enabled  
messages = chat_prompt  
    
# Generate the completion  
completion = client.chat.completions.create(  
    model=deployment,  
    messages=messages,  
    max_tokens=800,  
    temperature=0.7,  
    top_p=0.95,  
    frequency_penalty=0,  
    presence_penalty=0,  
    stop=None,  
    stream=False
)

print(completion.to_json())
# {
#   ...
#   "choices": [
#     {
#       "message": {
#         "content": "Hello! How can I assist you in finding information today?",
#         "role": "assistant"
#       },
#   ...
# }
```

Note that the response from the API is very rich:

```json
{
  "id": "chatcmpl-xxx",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "logprobs": null,
      "message": {
        "content": "Hello! How can I assist you in finding information today?",
        "role": "assistant"
      },
      "content_filter_results": {
        "hate": {
          "filtered": false,
          "severity": "safe"
        },
        "self_harm": {
          "filtered": false,
          "severity": "safe"
        },
        "sexual": {
          "filtered": false,
          "severity": "safe"
        },
        "violence": {
          "filtered": false,
          "severity": "safe"
        }
      }
    }
  ],
  "created": 1735936877,
  "model": "gpt-35-turbo",
  "object": "chat.completion",
  "system_fingerprint": null,
  "usage": {
    "completion_tokens": 12,
    "prompt_tokens": 19,
    "total_tokens": 31
  },
  "prompt_filter_results": [
    {
      "prompt_index": 0,
      "content_filter_results": {}
    }
  ]
}
```

### 1.2 Overview of LLMs

Summary of flow in an LLM:

- A text is a sequence of words.
- Words are converted into tokens: indices in a vocabulary.
- Tokens are converted into embedding vectors.
- An LLM is a probability machine: given the input sequence of words (tokens, vectors), the probabilities of all the vocabulary tokens to be the next are predicted. Then, we choose according to different strategies, e.g., we pick the most probable one, or sample among the most probable ones.

Benefits and risks of LLMs:

- They create coherent text.
- Tasks that work well:
  - Summarization, given the original text.
  - Rewriting texts, given all the information.
- **BUT, they hallucinate**: they basically predict the next word based on the probability distributions they have learned.

Mitigating the risks of LLMs, in increasing order of difficulty and cost:

- Prompt engineering
- Add retrieval and extend context (RAG)
- Fine-tune with new documents
- Train from scratch

![LLM Risk Mitigation](./assets/llm_risk_mitigation.png)

LLM Operations is composed of the following components:

1. Data
2. Models: training, fine-tuning, storage
3. Testing: evaluation of models
4. Search: we need to search data, etc.
5. Prompting: interface, commands to the LLMs
6. Deployment to the cloud: packaging, setting up inference, scaling, etc.
7. Monitoring: observe performance, drift, etc.

All those components need to be able to connect to each other and we need to be able to automate them.

Very interesting blog post by MS: [An Introduction to LLMOps: Operationalizing and Managing Large Language Models using Azure ML](https://techcommunity.microsoft.com/blog/machinelearningblog/an-introduction-to-llmops-operationalizing-and-managing-large-language-models-us/3910996).

### 1.3 LLM Deployment in Azure

:warning: If we get a deployment error associated with an *unregistred service*, that's because we need to **register** a service to our Subscription.
In Azure Portal:

    Select Subscriptio > Settings: Resource Providers > Search for Service, e.g.: Microsoft.MachineLearningServices > Register

#### Azure Machine Learning

The service Azure Machine Learning has many models from many vendors:

- All available models have been vetted to work on Azure.
- Some models might be available in given regions/zones.

Also, note that we can upload our models, e.g., in ONNX format!

Steps to browse the models:

- Create an instance/workspace, e.g., `demo-coursera-ml-us`.
- Open workspace and go to `Model catalog`.
- We can filter by many aspects:
  - Collections: HuggingFace, Meta, etc.
    - OpenAI models appear also here, but if we deploy one, we are redirected from Azure Machine Learning space to the Azure AI Foundry (previously Azure Open AI Studio).
  - Inference Task: text completion, summarization, object detection, etc.
  - Licenses
  - etc.

Important sections (left menu):

- `Model catalog`: curated models from Azure
- `Assets: Models`: all our model appear here, we can upload models, too
- `Assets: Environment`: a deployment requires an environment, which is usually a Docker image (`Dockerfile`) with optional files (`requirements.txt`); there are environments available and we can create custom ones.
- `Assets: Endpoints`: Models are deployed to an endpoint, which has a URI as well ass access credentials.
- `Manage: Compute`: Deployments require a compute resource, which is instantiated here.

Deployment options for the models:

- After a model is selected (e.g., from the Model catalog), we click on `Deploy`; we usually have the options
  - Pay-as-you-go
  - Real-time
  - Batch inference
- When we select a deployment method, sometimes we get the error *not enough quota*
  - That's because we need to first define a `Compute` instance 
    - We can select CPU/GPU
    - Look at the price per hour: the more resources, the more expensive
- If we want to deploy a custom model we uploaded, the procedure is similar, but we have to specify
  - The model (which needs to be uploaded, e.g. an ONNX file)
  - The environment (`Dockerfile` + `requirements.txt`)
  - The scoring script
  - The deployment configuration:
    - Managed compute vs Kubernetes
    - Authentication type
    - Timeout
    - etc.
  - etc.

Check [`02_azure_machine_learning_basics.ipynb`](./notebooks/02_azure_machine_learning_basics.ipynb) to see how a Pytorch model can be converted to ONNX format and uploaded to Azure Machine Learning.

Snapshot: Upload an ONNX Model

![Upload an ONNX Model](assets/azure_ml_upload_onnx_model.png)

Snapshot: Deploy an ONNX Model

![Deploy an ONNX Model](assets/azure_ml_deploy_onnx_model.png)

Snapshot: Scoring of an ONNX Model

![Scoring of an ONNX Model](assets/azure_ml_deploy_onnx_model_scoring_environment.png)

Snapshot: Deploy a Model from Catalog

![Deploy a Model from Catalog](assets/azure_ml_deploy_model_catalog.png)

Snapshot: Deploy Compute

![Deploy Compute](assets/azure_ml_deploy_compute.png)

Snapshot: Deployed Compute Details. Note that we can launch come applications that are already running!

![Deployed Compute Details](assets/azure_ml_compute_details.png)

#### Azure AI Content Safety

Check: [What is Azure AI Content Safety?](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/overview)

Azure AI Content Safety is an API which filters input/output content. Examples:

> - [Jailbreak risk detection](https://learn.microsoft.com/en-us/rest/api/contentsafety/text-operations/detect-text-jailbreak?view=rest-contentsafety-2024-02-15-preview&tabs=HTTP): Scans text for the risk of a User input attack on a Large Language Model. This is an input filter.
> - [Protected material test](https://learn.microsoft.com/en-us/rest/api/contentsafety/text-operations/detect-text-protected-material?view=rest-contentsafety-2024-09-01&tabs=HTTP): Scans AI-generated text for known text content (for example, song lyrics, articles, recipes, selected web content). This is an output filter.

**Content Safety** is a service, so we can create an instance in the Portal: `Search > Content safety`.

- When we instantiate/create one, we open the Azure Content Safety Studio
- We can configure and try the Content Safety functionality in the Azure Content Safety Studio.
  - However, we are going to use it usually via an API.
  - The Azure Content Safety Studio has links to the documentation and Python code examples where the API calls are performed via `resquests`.

#### Azure Open AI (Studio - or Azure AI Foundry)

Compared to Azure Machine Learning, the Azure OpenAI Studio (now Azure AI Foundry) is constrained to OpenAI models but it is much faster and easier to use.

See the previous section [Azure OpenAI](#azure-openai).

## 2. LLMs with Azure

### 2.1 Azure Machine Learning and LLMs

#### Check Quotas

Before deploying anything, we need to make sure we have the necessary quotas active.

    Azure Portal > Quotas > Machine Learning: Find Compute resource we need in region
        Device quota not enough? Request increase

The resource we are going to use the quota with needs to be in the same region; in our case, our Azure Machine Learning workspace must be deployed in the same region, e.g., East US, West Europe, etc.

A typical compute quota for ML applications is an NCAS device: NVIDIA GPU-enabled virtual machine (VM); they are expensive, so choose wisely. Example choice:

`Standard NCASv3_T4 Family Cluster Dedicated vCPUs` in West Europe (Usage: 0 of 12 available).

If we request it a quota increase, it takes some minutes to process.

![Quotas](./assets/quotas.png)

Also, note that the Azure Machine Learning Studio has also a section/tab called *Quotas*:

![Quotas in Azure Machine Learning](./assets/quotas_azure_ml.png)

Some additional device families:

- Standard DASv4 Family Cluster Dedicated vCPUs: no GPU, but CPU and memory intensive work; e.g., for classical ML.
- Standard EDSv4 Family Cluster Dedicated vCPUs: no GPU, but large amounts of memory per CPU (AMD processors); e.g., for large-scale enterprise applications.
- Standard ESv3 Family Cluster Dedicated vCPUs: no GPU, memory-intensive workloads (Intel processors); e.g., Spark applications.
- ...

More information: [Azure subscription and service limits, quotas, and constraints](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/azure-subscription-service-limits).

#### Create Compute Instances

Once we have checked that we have available quota, we need to deploy a compute instance.

In out Azure Machine Learning Workspace (recall it's tied to a region), we go to `Compute` menu and deploy a compute instance:

- Select GPU if we need
- Select device; e.g.: Standard_NC6s_v3 (6 cores, 112 GB RAM, 736 GB disk)
  - GPU - 1 x NVIDIA Tesla V100
  - USD 3.82/hr (when running)
- After we deploy it and set it running, when we click on it 
  - We see we can apply some actions: Stop, Delete, Restart, etc.
  - We have some running applications available and accessible via link: Jupyter, etc.

![Azure Machine Learning: Compute Deployment](./assets/azure_ml_compute_deployed_1.png)

![Azure Machine Learning: Compute Deployment](./assets/azure_ml_compute_deployed_2.png)

More information: [Tutorial: Create resources you need to get started](https://learn.microsoft.com/en-gb/azure/machine-learning/quickstart-create-resources?view=azureml-api-2).

#### Deploy a Model

When a Compute instance is deployed on our Azure Machine Learning Workspace, we can select a LLM model from the `Model catalog` and deploy it.

:warning: The example chosen in the course was `microsoft-phi-2`; however, I was not able to replicate the deployment, even tough I followed all the steps. I think the error is related to the quotas, and the current high demand for GPUs.

In the course video, the instructor shows that once deployed, we have several infos on the resource:

- Swagger URI: API documentation
- Authentication configuration: key or token
- REST endpoint
- Also, there as some important tabs:
  - Test
  - Consume: information for using the endpoint
    - Keys
    - Endpoint URL
    - Code examples
  - Monitoring
  - Logs tab

![Azure Machine Learning: Model Deployment](./assets/azure_ml_model_deployment.png)

### 2.2 Azure OpenAI Service

Some months ago we needed to request access to Azure OpenAI; but apparently that's not the case anymore.

Compared to Azure Machine Learning, the Azure OpenAI Studio (now Azure AI Foundry) is constrained to OpenAI models but it is much faster and easier to use.

See the previous section [Azure OpenAI](#azure-openai).

### 2.3 Azure OpenAI APIs

This section is the continuation of the section [Azure OpenAI](#azure-openai).

Following the code example from the playground, a mini app is build.

- The application maintains a state called `history`, which contains the conversation so far.
- With a `while True` loop, we interact with the deployed LLM while the conversation is kept in memory (context).
- The code is in [`01_azure_open_ai_basics.ipynb`](./notebooks/01_azure_open_ai_basics.ipynb), but it is better suited to be run in a script.

```python
import os  
import base64
from openai import AzureOpenAI
from openai import ChatCompletion
from dotenv import load_dotenv
from typing import List, Dict, Optional

load_dotenv(override=True, dotenv_path=".env")

# Now, the URL is shorter and we add the endpoint and deployment name
endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

# Initialize Azure OpenAI client with key-based authentication    
client = AzureOpenAI(  
    azure_endpoint=endpoint,  
    api_key=subscription_key,  
    api_version="2024-05-01-preview",  
)

def chat(user_message: str, history: Optional[list[dict]] = None) -> ChatCompletion:
    if history is None:
        history = [
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find information."
            }
        ]

    # Insert the user message into the history
    history.append(
        {
            "role": "user",
            "content": user_message
        }
    )
    
    # Generate the completion  
    completion = client.chat.completions.create(  
        model=deployment,  
        messages=history,  
        max_tokens=800,  
        temperature=0.7,  
        top_p=0.95,
        frequency_penalty=0,  
        presence_penalty=0,  
        stop=None,  
        stream=False
    )
    
    # Extend history
    history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content
        }
    )
    
    return completion, history

history = None
while True:
    user_message = input(">>: ")
    completion, history = chat(user_message, history)
    print(f"assistant: {completion.to_dict()['choices'][0]['message']['content']}")

```

## 3. Extending with Functions and Plugins

### 3.1 Improved Prompts with Semantic Kernel

TBD.

### 3.2 Extending Results with Functions

TBD.

### 3.3 Using Functions with External APIs

TBD.

## 4. Building an End-to-End Application in Azure

### 4.1 Architecture

Summary:

- The section covers the architectural overview and components involved in building an end-to-end application using large language models; specifically, the RAG (Retrieval Augmented Generation) pattern is introduced.
- The application is built entirely using Microsoft Azure Cloud services; it uses, among others, the Azure AI Search service.
- Github Actions are also introduced, as a way to automate deployment.

LLMs are trained in many tokens/texts, but they might not have specific knowledge we want. A [RAG (Retrieval Augmented Generation)](https://arxiv.org/abs/2005.11401) application is able to plug in new knowledge to the model via its context:

- First, we index documents which contain new information; e.g., in a vector DB.
- When we ask the Query, the Framework (e.g., LangChain or our Application) retrieves the Documents which are likely to contain the answer.
- With those Documents, the Context is created.
- The LLM is invoked in a Prompt which contains the Query and the Context (i.e., we already provide the answer as specified in our Documents).
- The answer of the LLM is returned to the User.

![RAG Overview](./assets/rag_overview.png)

Azure enables the necessary components to implement [Retrieval Augmented Generation (RAG) using Azure AI Search](https://learn.microsoft.com/en-us/azure/search/retrieval-augmented-generation-overview).

![Azure RAG](./assets/azure_rag.png)

#### Azure AI Search

Azure AI Search (previously known as Azure Cognitive Search) is a basic Azure service we need; it allows to search data from storage components.

We can manually create it, e.g.:

![Create an Azure AI Search Service](./assets/create_azure_ai_search.png)

Important: Choose the Free Tier with 50 MB of storage.

Once created and configured, if we go to the resource in the Portal, we see we can

- **Import Data** manually
  - For that, we need to either create or connect a Blob Storage Account
- We can search in the **Indexes**
- etc.

![Azure AI Search Service](./assets/azure_ai_search_service.png)

More information can be found in the official Azure documentation: [Create an Azure AI Search service in the Azure portal](https://learn.microsoft.com/en-gb/azure/search/search-create-service-portal)

- **Configure authentication** (for programmatic access)
- Scale your service (Partitions and Replicas)
- When to add a second service
- Add more services to a subscription

#### Github Actions

Github Actions allow to easily deploy an application to the cloud, i.e., Azure.

- We can enter secrets to Github.
- Under `.github/workflows` a YAML file is created with all the jobs/steps necessary for deployment.
- We usually get a template YAML which we modify to our needs.
- Typical steps:
  - Checkout repo branch
  - Setup Docker
  - Login to Github and get secrets
  - Build our container and push it to Azure Container Registry (ACR)
  - Deploy Azure Container App (after authentication)

The example shown in the video is the following:

[`alfredodeza/huggingface-azure-acr/.github/workflows/main.yml`](https://github.com/alfredodeza/huggingface-azure-acr/blob/main/.github/workflows/main.yml)

```yaml
name: Trigger auto deployment for demo-container

#env:
#  AZURE_CONTAINER_APP_NAME: demo-container
#  AZURE_GROUP_NAME: demo-container

# When this action will be executed
on:
  # Automatically trigger it when detected changes in repo. Remove comments to enable
  #push:
  #  branches: 
  #    [ main ]

  # Allow mannually trigger 
  workflow_dispatch:      

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout to the branch
        uses: actions/checkout@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v1

      - name: Log in to GitHub container registry
        uses: docker/login-action@v1.10.0
        with:
          registry: demoalfredo.azurecr.io
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}

      - name: Lowercase the repo name and username
        run: echo "REPO=${GITHUB_REPOSITORY,,}" >>${GITHUB_ENV}

      - name: Build and push container image to ACR registry
        uses: docker/build-push-action@v2
        with:
          push: true
          tags: demoalfredo.azurecr.io/${{ env.REPO }}:${{ github.sha }}
          file: ./Dockerfile

  deploy:
    runs-on: ubuntu-latest
    needs: build
    
    steps:
      - name: Azure Login
        uses: azure/login@v1
        with:
          creds: ${{ secrets.AZURE_CREDENTIALS }}


      - name: Deploy to containerapp
        uses: azure/CLI@v1
        with:
          inlineScript: |
            az config set extension.use_dynamic_install=yes_without_prompt
            az containerapp registry set -n demo-container -g demo-container --server demoalfredo.azurecr.io --username ${{ secrets.ACR_USERNAME }} --password ${{ secrets.ACR_PASSWORD }}
            az containerapp update -n demo-container -g demo-container --cpu 2 --memory 4Gi
            az containerapp update -n demo-container -g demo-container --image demoalfredo.azurecr.io/alfredodeza/huggingface-azure-acr:${{ github.sha }}
```

#### Azure AI Document Intelligence

A link to this blog post is provided:

[Elevating RAG and Search: The Synergy of Azure AI Document Intelligence and Azure OpenAI](https://techcommunity.microsoft.com/blog/azure-ai-services-blog/elevating-rag-and-search-the-synergy-of-azure-ai-document-intelligence-and-azure/4006348)

Key ideas in the blog post:

> The Azure AI Document Intelligence Layout model offers a comprehensive solution for semantic chunking by providing advanced content extraction and document structure analysis capabilities.
> With this model, you can easily extract paragraphs, tables, titles, section headings, selection marks, font/style, key-value pairs, math formulas, QR code/barcode and more from various document types.
> The extracted information can be conveniently outputted to markdown format, enabling you to define your semantic chunking strategy based on the provided building blocks.
> The model is highly scalable in Optical Character Recognition (OCR), table extraction, document structure analysis (e.g., paragraphs, titles, section headings), and reading order detection, ensuring high-quality results driven by AI capabilities.
> It supports 309 printed and 12 handwritten languages.

We can go to [Azure AI Document Intelligence Studio](https://documentintelligence.ai.azure.com/studio/layout), choose the Analyze options and try it!

Additional resources:

- [Quickstart: Document Intelligence SDKs](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/quickstarts/get-started-sdks-rest-api?view=doc-intel-4.0.0&pivots=programming-language-python#layout-model) – use your preferred SDK or REST API to extract content and structure from documents.
- [Sample code of using Layout API to output in markdown format](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/documentintelligence/azure-ai-documentintelligence/samples/sample_analyze_documents_output_in_markdown.py).

#### Extra Exercises

Azure AI Search:

- [ ] Use the Azure CLI or SDK to create an Azure AI Search service.
- [ ] Configure an Azure role like "Search Service Contributor" for managing the service.
- [ ] Try creating an index and loading data using the portal import data wizard.
- [ ] Experiment with different partition and replica configurations.
- [ ] Compare capabilities between the Free and Basic tiers.

Azure Document Intelligence:

- [ ] Analyze a document via the portal.
- [ ] Convert a document into Markdown with the SDK.

### 4.2 RAG with Azure AI Search

TBD.

### 4.3 Deployment and Scaling with Github Action

TBD.
