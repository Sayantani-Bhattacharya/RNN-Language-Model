# RNN-Language-Model
MSAI-437 Homework 3

## Architecture and Design choices:
The model is a simple Recurrent Neural Network (RNN) designed for a language modeling task, where the objective is to predict the next token in a sequence of tokens (words, characters, etc.). The architecture consists of the following key components:
1. Embedding Layer:
- Purpose: This layer converts token indices into dense vectors (embeddings) of fixed dimension (embedding_dim). This helps in learning richer representations for each word.
- Dimension: The embedding layer is of size (vocab_size, embedding_dim), where vocab_size is the total number of unique tokens in the vocabulary, and embedding_dim is the number of dimensions each token is mapped to.
- Design Decision: The use of an embedding layer helps reduce the dimensionality of the input data (which is typically sparse) and makes it more suitable for neural network processing.
  
2. RNN Layer:
- Purpose: The RNN processes the sequence of embeddings (one token at a time) and generates hidden states that encode information about the sequence seen so far.
- Single Layer RNN: The model uses a single-layer RNN (num_layers=1). While deeper RNNs (multiple layers) can capture more complex patterns, a single layer is simpler and might perform adequately for some language modeling tasks.
- Hidden Size: The hidden state dimension (hidden_dim) is 256, which balances between model complexity and computational efficiency.

3. Dropout Layer:
- Purpose: Dropout is used as a regularization technique to prevent overfitting by randomly setting a fraction of the input units to zero during training.
Inapplicability for Single Layer RNNs: Although dropout is only effective when num_layers > 1, it is included for experimentation to observe any effects when training with a single layer.
- Design Decision: This is a common practice in neural networks to improve generalization, though it won’t have a significant effect in the current architecture.

4. Fully Connected (FC) Layer:
- Purpose: After the RNN processes the input, the final hidden states are passed through a fully connected layer to generate logits, which represent the predicted scores for each token in the vocabulary.
- Shape of Output: The output shape of the logits is (batch_size, seq_length, vocab_size), which corresponds to the predicted scores for each token at each time step in the sequence.

5. Hidden State Initialization:
- Purpose: The hidden state of the RNN is initialized as a tensor of zeros before each forward pass. This ensures that the model starts with no prior knowledge of the input sequence.

  
## Hyper-parameters:
1. embedding_dim = 100
- Reasoning: The embedding dimension controls the size of the vector representation for each token. A value of 100 is a common choice, balancing between capturing semantic relationships and model efficiency. Larger values might improve model capacity, but at the cost of increased computational overhead.

2. hidden_dim = 256
- Reasoning: The hidden dimension is the size of the hidden state in the RNN. A value of 256 provides a reasonable tradeoff between the model’s ability to capture dependencies and its computational complexity. Higher values might improve performance but could lead to overfitting or slower training.

3. dropout = 0.5
- Reasoning: Dropout rate of 0.5 is commonly used as it helps in regularization. This means 50% of the activations are randomly dropped during training, preventing the network from becoming too reliant on certain neurons. While dropout is not effective for single-layer RNNs, it's kept here for experimentation and generalization purposes.

4. vocab_size: 10001
- Reasoning: A larger vocabulary allows for more diverse token representations, but it increases the complexity of the model. This value is chosen based on the dataset and needs to be set appropriately for the application.


## Learning curves of perplexity vs. epoch on the training and validation sets:

## Final test set perplexity

## Improvement stategies:

## Points to Note:

1. Extract the vocabulary only from the training set. Since the model learns from the training data, we should determine the most frequent words only from train.txt. The validation and test sets should not influence vocabulary selection.
2. Thus we must convert words to their corresponding integer indices (using the vocabulary from the training set). Replace any out-of-vocabulary (OOV) words in validation and test sets with <unk>.
