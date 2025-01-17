# Sentence Embedding and t-SNE Visualization using DistilBERT

This repository contains a Python script that demonstrates how to use the pre-trained DistilBERT model to generate sentence embeddings and visualize these embeddings using t-SNE (t-distributed Stochastic Neighbor Embedding). 

## Table of Contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Usage](#usage)
    - [Loading the Model](#loading-the-model)
    - [Generating Sentence Embeddings](#generating-sentence-embeddings)
    - [Visualizing Embeddings with t-SNE](#visualizing-embeddings-with-t-sne)
4. [Example Output](#example-output)
5. [Conclusion](#conclusion)

## Introduction

t-SNE is a powerful tool for visualizing high-dimensional data in a 2D space, making it easier to identify patterns and relationships. By using pre-trained DistilBERT, we can generate high-quality, context-aware sentence embeddings, which are then projected onto a 2D plane using t-SNE.

## Requirements

To run the script, you need the following libraries installed:

- `transformers`
- `torch`
- `scikit-learn`
- `matplotlib`
- `pandas`   (optional)
- `numpy`

You can install the required libraries using pip:

```bash
pip install transformers torch scikit-learn matplotlib pandas numpy
```

## Usage

### Loading the Model

The script loads the pre-trained DistilBERT model and tokenizer using the `transformers` library:

```python
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model= DistilBertModel.from_pretrained('distilbert-base-uncased')
    return tokenizer, model
```

### Generating Sentence Embeddings

Sentence embeddings are generated using the pre-trained DistilBERT model. The CLS token is used as the sentence embedding:

```python
def embedding(sentences, tokenizer, model):
    embeddings= []
    with torch.no_grad():
        for s in sentences:
            inputs = tokenizer(s, return_tensors='pt', truncation= True, padding = True, max_length=128)
            outputs = model(**inputs)
            sentence_embedding= outputs.last_hidden_state[:,0,:].squeeze().numpy()
            embeddings.append(sentence_embedding)
    return np.array(embeddings)
```

### Visualizing Embeddings with t-SNE

The t-SNE algorithm is applied to reduce the dimensionality of the sentence embeddings and visualize them in a 2D space:

```python
def plot(embedding, labels=None):
    tsne=TSNE(n_components=2, perplexity=5, random_state=42)
    reduced_embeddings= tsne.fit_transform(embedding)
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1],c='blue', label='Sentences')
    if labels:
        for i, label in enumerate(labels):
            plt.text(reduced_embeddings[i,0]+ 0.1, reduced_embeddings[i, 1] + 0.1, label, fontsize=9)
    plt.title("t-SNE Visualization of Sentence Embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend()
    plt.show()
```

### Running the Script

To run the script, call the `main()` function:

```python
def main():
    sentences=[
        "I am a computer science student",
        "I like Natural language processing",
        "NLP is a subfield of artificial intelligence",
        "Artificial intelligence is a trending field",
        "That mall is too big",
        "Data science is a growing field.",
        "I enjoy reading books.",
        "The food was good.",
        "Traveling is a great way to learn things.",
        "BERT is a powerful model for NLP tasks."
    ]
    tokenizer, model = load_model()
    print("Loading DistilBERT model...")
    embeddings = embedding(sentences, tokenizer, model)
    print("Generating t-SNE visualization...")
    plot(embeddings, labels=sentences)

if __name__ == "__main__":
    main()
```

## Example Output

When you run the script, it will generate a t-SNE plot visualizing the sentence embeddings in a 2D space. The plot will show how sentences with similar contexts or themes are grouped together, making it easier to identify patterns in the data.

## Conclusion

This project demonstrates the use of DistilBERT for generating high-quality sentence embeddings and visualizing them using t-SNE. Feel free to explore the script and modify it for your own text analysis and visualization needs.

---
