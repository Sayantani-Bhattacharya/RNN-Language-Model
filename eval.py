# This file loads the trained model and evaluates it on the test set by computing the loss and perplexity.
# Ensure the hyperparameters (embedding dimension, hidden dimension, dropout, and vocab_size) match those used in train.py!

# eval.py
import torch
import torch.nn as nn
from model import RNNLanguageModel

# Load processed data to get the test loader and vocabulary size.
data = torch.load("processed_data.pth", weights_only=False)
test_loader = data['test_loader']
# vocab_size = data['vocab_size']
vocab_size = data.get('vocab_size', 10001)

# Hyperparameters (should be identical to training).
embedding_dim = 100
hidden_dim = 256
dropout = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load the saved weights.
model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, dropout).to(device)
model.load_state_dict(torch.load("rnn_language_model.pth", map_location=device))
model.eval()

criterion = nn.CrossEntropyLoss()

total_loss = 0.0
hidden = None

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        if hidden is None or hidden.size(1) != batch_size:
            hidden = model.init_hidden(batch_size, device)
        else:
            hidden = hidden.detach()
        outputs, hidden = model(inputs, hidden)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        total_loss += loss.item()

avg_test_loss = total_loss / len(test_loader)
test_perplexity = torch.exp(torch.tensor(avg_test_loss))
print(f"Test Loss: {avg_test_loss:.4f} - Test Perplexity: {test_perplexity:.2f}")

