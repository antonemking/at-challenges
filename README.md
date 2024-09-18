# 🚩 Challenge #0: Building Attention

### Introduction

In this first challenge, we will start from the basics and build our way up toward understanding the foundational components of the attention mechanism. By the end of this challenge, you will implement a simplified version of the attention mechanism, laying the groundwork for more complex parts of the Transformer model in later challenges.

# 🚀 Let's Get Started!

Welcome to Challenge 1 of your journey to master "Attention Is All You Need"! 🎮 In this challenge, you will code, experiment, and reason your way through the core building blocks of the attention mechanism. Let’s break it down into smaller, digestible tasks that will help you grasp the heart of the Transformer model.

## Attention Mechanism

The scaled dot-product attention is computed as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:
- \( Q \) is the Queries matrix.
- \( K \) is the Keys matrix.
- \( V \) is the Values matrix.
- \( d_k \) is the dimension of the queries and keys.
- \( QK^T \) is the dot product between the queries and keys.
- \( $$\text{softmax} \$$ ) is applied to the scaled dot product.


### Part 1 

We’ll start by understanding the dimensions that make up the attention mechanism.
The basic unit is the embedding, which represents the data (e.g., tokens in a sentence) in a numerical format that the model can process.

💡 Exercise 1: Defining Key Parameters
In your notebook, define the key parameters for our simplified attention mechanism:

- __d_model__: The size of the embeddings (number of features per token).
- __seq_len__: The number of tokens in a sequence (e.g., words in a sentence).
- __batch_size__: The number of sequences processed at once.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonemking/at-challenges/blob/challenge-0/at_sr_challenge0.ipynb?authuser=0&copy=true&t=latest)

💡 Why do you think we chose d_model = 64?
*Hint: Think about computational efficiency. Would increasing this value make the model slower or faster? How does it affect the model’s capacity to learn complex patterns?*

### Part 2: Generating Queries, Keys, and Values

💡 Exercise 2: Generating Random Queries, Keys, and Values
In your notebook, Let’s generate these matrices for each token in our sequence:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonemking/at-challenges/blob/challenge-0/at_sr_challenge0.ipynb?authuser=0&copy=true&t=latest)

💡 Why do queries, keys, and values have the same dimensions?
*Hint: Think about how we use dot products to compare each query with every key.*

### Part 3: Computing Attention Scores
The next step is to compute attention scores. These scores represent how much focus each token should have on the others. The higher the score, the more focus a token has.

💡 Exercise 3: Dot Product of Queries and Keys
The attention scores are calculated by taking the dot product of the queries and keys.

Let’s compute the attention scores in your notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonemking/at-challenges/blob/challenge-0/at_sr_challenge0.ipynb?authuser=0&copy=true&t=latest)

💡 The result of the dot product is a matrix of shape (batch_size, seq_len, seq_len). Why?
*Hint: How many comparisons are made between each query and the keys?*

### Prerequisites

Before starting this challenge, you should have:

- Basic proficiency in Python programming.
- Understanding of fundamental neural network concepts.
- Familiarity with machine learning frameworks like PyTorch or TensorFlow is a plus but not required.

Open the Google Colab Notebook [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antonemking/at-challenges/blob/challenge-0/at_sr_challenge0.ipynb?authuser=0&copy=true&t=latest)
