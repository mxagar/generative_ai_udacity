# Project: Applying Lightweight Fine-Tuning to a Foundation Model

See the [5th section of `../README.md`](../README.md#5-project-applying-lightweight-fine-tuning-to-a-foundation-model).

This folder contains the notebook and material related to the learning part of the project.

The notebook is very important, since it summarizes the complete module/section: [`peft_howto.ipynb`](./peft_howto.ipynb)

Here's a summary of what is being done in the notebook **`peft_howto.ipynb`**, based on its content:

- A text classification task is defined using both a custom and Hugging Face dataset, specifically a spam detection dataset of SMS messages.
- The dataset is explored through basic EDA, including duplicate and empty message analysis, message length distributions and hidden state or feature analysis using `UMAP` projections.
- Tokenization is performed using a `DistilBertTokenizerFast`, adding `input_ids` and `attention_mask` to the dataset.
- A PEFT (Parameter-Efficient Fine-Tuning) LoRA adapter is added on top of a `DistilBertForSequenceClassification` model to reduce training complexity.
- Training is done using Hugging Face’s `Trainer` with metrics like accuracy, precision, recall, and F1, along with logging to TensorBoard.
- A `compute_metrics` function is implemented to evaluate predictions and also log the loss during evaluation.
- Predictions are made using the `Trainer`’s `.predict()` and a custom `predict()` function for single inference cases.
- After training, the LoRA adapters are merged into the base model and the combined model is saved.
- The merged model is exported to ONNX format, enabling lightweight and hardware-agnostic deployment.
- An ONNX-based evaluation pipeline is created using `onnxruntime`, and metrics (accuracy, precision, recall, F1) are computed on test data.

The final project itself is based on that notebook and can be found here: [mxagar/llm_peft_fine_tuning_example](https://github.com/mxagar/llm_peft_fine_tuning_example).
