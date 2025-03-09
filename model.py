# It uses an embedding layer of dimension 100, 
# a single-layer RNN, and a fully connected output layer.
# dropout as a hyper-parameter—even though dropout won’t apply when num_layers=1, it’s kept for experimentation,
# and a linear layer that maps the hidden state to vocabulary scores.

import torch
import torch.nn as nn

class RNNLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, dropout=0.5):
        super(RNNLanguageModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Embedding layer: maps token indices to embedding vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN layer: a single-layer RNN.
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=1,
                          dropout=dropout, batch_first=True)
        # Dropout layer (note: dropout in RNN is effective only for num_layers>1)
        # self.dropout = nn.Dropout(dropout)                                       ######################3 commented this.
        # Fully connected layer: maps RNN output to vocabulary size.
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        # Obtain embeddings for input tokens.
        emb = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        # Pass through the RNN.
        output, hidden = self.rnn(emb, hidden)  # output: (batch_size, seq_length, hidden_dim)
        # Apply dropout.
        # output = self.dropout(output)  ##########################################################3 commented this.
        # Get logits for each token.
        logits = self.fc(output)  # (batch_size, seq_length, vocab_size)
        return logits, hidden

    def init_hidden(self, batch_size, device):
        # Initialize hidden state with zeros.
        # For an RNN with one layer, the hidden state shape is (1, batch_size, hidden_dim)
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
