# 🚀 Challenge 0: Building Attention

## Introduction

Welcome to the first challenge in mastering "Attention Is All You Need"! In this challenge, we'll build our understanding of the attention mechanism from the ground up. By the end, you'll have implemented a simplified version of the attention mechanism, setting the stage for more complex components of the Transformer model in future challenges.

## 📚 Background: The Attention Mechanism

The core of the Transformer model is the scaled dot-product attention, computed as:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V $$

Where:
- Q is the Queries matrix
- K is the Keys matrix
- V is the Values matrix
- d_k is the dimension of the queries and keys

Let's break this down step by step!

## 🎯 Challenge Steps

### Step 1: Define Key Parameters

Before we dive into the code, let's understand the key parameters that define our attention mechanism.

📘 **Learn:**
- `d_model`: The size of our embeddings (number of features per token)
- `seq_len`: The number of tokens in a sequence (e.g., words in a sentence)
- `batch_size`: The number of sequences processed at once

🤔 **Think:** Why might we choose `d_model = 64`? Consider computational efficiency and the model's capacity to learn.

✍️ **Code:** Define these parameters in a Colab notebook.

[Open Colab for Step 1](https://colab.research.google.com/drive/your-notebook-link-here)

### Step 2: Generate Queries, Keys, and Values

Now that we have our parameters, let's generate the building blocks of attention: Queries, Keys, and Values.

📘 **Learn:**
- Queries, Keys, and Values are generated for each token in our sequence
- They all have the same dimensions initially

🤔 **Think:** Why do Queries, Keys, and Values have the same dimensions? How does this relate to the dot product operation?

✍️ **Code:** Generate random Queries, Keys, and Values matrices.

[Open Colab for Step 2](https://colab.research.google.com/drive/your-notebook-link-here)

### Step 3: Compute Attention Scores

With our Queries, Keys, and Values in hand, we can now compute the attention scores.

📘 **Learn:**
- Attention scores represent how much focus each token should have on others
- They're computed using the dot product between Queries and Keys

🤔 **Think:** What does the shape of the resulting attention score matrix tell us about the relationships between tokens?

✍️ **Code:** Implement the dot product between Queries and Keys to compute attention scores.

[Open Colab for Step 3](https://colab.research.google.com/drive/your-notebook-link-here)

### Step 4: Apply Scaling and Softmax

To complete our attention mechanism, we need to scale our scores and apply the softmax function.

📘 **Learn:**
- Scaling helps manage large values in dot products of high-dimensional vectors
- Softmax turns our scores into a probability distribution

🤔 **Think:** Why do we scale by $\sqrt{d_k}$? How does this affect the gradients during training?

✍️ **Code:** Implement scaling and softmax on the attention scores.

[Open Colab for Step 4](https://colab.research.google.com/drive/your-notebook-link-here)

### Step 5: Compute the Final Attention Output

The last step is to use our attention weights to compute a weighted sum of the Values.

📘 **Learn:**
- The final output is a combination of Values, weighted by attention scores
- This output represents the "attended" information for each token

🤔 **Think:** How does this weighted sum allow each token to "pay attention" to relevant parts of the sequence?

✍️ **Code:** Implement the final matrix multiplication to get the attention output.

[Open Colab for Step 5](https://colab.research.google.com/drive/your-notebook-link-here)

## 🎉 Conclusion

Congratulations! You've built a basic attention mechanism from scratch. This forms the foundation of the Transformer model. In the next challenge, we'll explore how to use this mechanism in a multi-head attention setup.

🧠 **Final Thought:** How might this attention mechanism be used in different types of sequences (e.g., text, time series, images)?

## 📚 Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - A visual guide to Transformers
- [Transformers from Scratch](https://e2eml.school/transformers.html) - A detailed walkthrough of Transformer implementation
