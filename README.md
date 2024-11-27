# Finetuning-GPT-and-Knowledge-Distillation
This repository contains the implementation of two techniques for improving the efficiency and performance of Large Language Models (LLMs):

- **Low-Rank Adaptation (LoRA)**
- **Knowledge Distillation using Recurrent Neural Networks (RNNs)**

Both methods are applied to the **Corpus of Linguistic Acceptability (CoLA)** dataset to evaluate their effectiveness.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
  - [Data Format](#data-format)
- [Problem Statements](#problem-statements)
  - [Problem 0: Text Generation](#problem-0-text-generation)
  - [Problem 1: Low-Rank Adaptation (LoRA)](#problem-1-low-rank-adaptation-lora)
  - [Problem 2: Knowledge Distillation](#problem-2-knowledge-distillation)
- [Installation](#installation)
- [Usage](#usage)
  - [Text Generation](#text-generation)
  - [LoRA Fine-Tuning](#lora-fine-tuning)
  - [Knowledge Distillation](#knowledge-distillation)
- [Results](#results)
  - [LoRA Results](#lora-results)
  - [Knowledge Distillation Results](#knowledge-distillation-results)
- [References](#references)

## Introduction

This project explores two advanced techniques to enhance the efficiency and performance of Large Language Models (LLMs):

1. **Low-Rank Adaptation (LoRA):** A parameter-efficient fine-tuning method that reduces the number of trainable parameters by injecting low-rank decomposition matrices into the pre-trained model layers.
2. **Knowledge Distillation:** A technique to transfer knowledge from a large, complex model (teacher) to a smaller, simpler model (student) to achieve a more efficient model without significant loss in performance.

These methods are implemented and evaluated on the **CoLA** dataset for the task of grammatical acceptability classification.

## Dataset

### The Corpus of Linguistic Acceptability (CoLA)

The CoLA dataset is a collection of English sentences labeled for grammatical acceptability. It is commonly used to evaluate language models on their understanding of linguistic rules.

- **Total Sentences:** 10,657 from 23 linguistics publications.
- **Splits:**
  - In-domain Train: 8,551 sentences (`train.tsv`)
  - In-domain Dev: 527 sentences (`dev.tsv`)
  - Out-of-domain Dev: 516 sentences

### Data Format

Each line in the dataset files is tab-separated with the following columns:

1. **Source:** Unique identifier for each sentence (numerical or alphanumeric code).
2. **Acceptability Label:** Binary indicator (`0` for unacceptable, `1` for acceptable).
3. **Sentence:** The sentence text.

Example:

```
S1    1    The quick brown fox jumps over the lazy dog.
```

## Problem Statements

### Problem 0: Text Generation

As a preliminary step, we use the base GPT-2 model to generate text continuations from a given prompt to ensure that the model and environment are set up correctly.

#### Prompt

```
"Grant me the serenity to accept the things I cannot change, the courage to change the things I can, and the wisdom to know the difference."
```

### Problem 1: Low-Rank Adaptation (LoRA)

LoRA reduces the number of trainable parameters by injecting trainable low-rank decomposition matrices into each layer of the Transformer architecture, while freezing the original pre-trained weights.

#### Key Concepts

- **Auxiliary Linear Layers:** Additional linear layers added in parallel to existing ones.
- **Low-Rank Update Storage:** Only the difference between pre-trained and fine-tuned weights is stored, assuming it's of low rank.
- **Decomposition with \( L \) and \( R \) Matrices:** The low-rank difference is represented using these two matrices.

#### Implementation Details

- **LoRALinear Class:** Implements the LoRA model by decomposing weight updates into low-rank matrices \( L \) and \( R \).
  - **Initialization:**
    - **\( U \):** Initialized using Kaiming Uniform initialization.
    - **\( V \):** Initialized to zeros.
  - **Scaling Factor (\( \alpha \)):** Controls the magnitude of the decomposition, calculated as \( \text{alpha} / \text{rank} \).

### Problem 2: Knowledge Distillation

Knowledge Distillation transfers knowledge from a large, complex model (teacher) to a smaller, simpler model (student), aiming to preserve performance while improving efficiency.

#### Model Architecture: DistilRNN

A simple Recurrent Neural Network (RNN) model used as the student in the distillation process.

- **Components:**
  - **Embedding Layer:** Converts input tokens into dense vectors.
  - **RNN Layer:** Captures temporal dependencies in the input sequences.
  - **Fully Connected Layer:** Outputs logits for classification.

## Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch (latest stable version recommended)
- Transformers library from Hugging Face
- CUDA-compatible GPU (optional but recommended)

### Clone the Repository

```bash
git clone https://github.com/yourusername/e0270-assignment2.git
cd e0270-assignment2
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*Note: Ensure that PyTorch is installed with CUDA support if using a GPU.*

## Usage

### Text Generation

To generate text using the base GPT-2 model:

```bash
python run.py gen <Your5DigitSR>
```

Replace `<Your5DigitSR>` with your 5-digit SR number.

### LoRA Fine-Tuning

To fine-tune the GPT-2 model using LoRA on the CoLA dataset:

```bash
python run.py LoRA <Your5DigitSR>
```

#### Hyperparameters

- **Learning Rate:** Adjust in `args.lr`
- **Batch Size:** Adjust in `args.batch_size`
- **Epochs:** Adjust in `args.epochs`
- **LoRA Rank:** Adjust in `args.LoRA_rank`

### Knowledge Distillation

To perform knowledge distillation from the LoRA fine-tuned GPT-2 model to the DistilRNN model:

```bash
python run.py distil <Your5DigitSR>
```

*Ensure that the LoRA fine-tuned model is saved and accessible at the specified `model_path`.*

## Results

### LoRA Results

- **Training Accuracy:** 95.44%
- **Validation Accuracy:** 80.83% at the end of the 9th epoch.
- **Number of Trainable Parameters:** Reduced by 99.5%, from 125.03 million to 0.63 million.

#### Loss and Accuracy Plot

![LoRA Training Plot](plots/lora_training_plot.png)

*The plot shows the training and validation loss decreasing over epochs, with corresponding increases in accuracy.*

### Knowledge Distillation Results

- **Training Accuracy:** 75%
- **Validation Accuracy:** 69.26% at the end of the 3rd epoch.
- **Number of Parameters in DistilRNN:** Significantly fewer than the GPT-2 model.

#### Loss and Accuracy Plot

![DistilRNN Training Plot](plots/distilrnn_training_plot.png)

*The plot indicates that the student model effectively learns from the teacher, though with some loss in accuracy due to reduced capacity.*

## References

- Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models." *arXiv preprint arXiv:2106.09685* (2021). [Link](https://arxiv.org/abs/2106.09685)
- Hinton, Geoffrey, Oriol Vinyals, and Jeff Dean. "Distilling the knowledge in a neural network." *arXiv preprint arXiv:1503.02531* (2015). [Link](https://arxiv.org/abs/1503.02531)
- CoLA Dataset: [Link](https://nyu-mll.github.io/CoLA/)
- Papers with Code - Linguistic Acceptability on CoLA: [Link](https://paperswithcode.com/sota/linguistic-acceptability-on-cola)

---
