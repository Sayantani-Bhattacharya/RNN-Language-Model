# Loads processed data, initializes the model,
# and trains it for 20 epochs using the Adam optimizer and cross-entropy loss. 
# It also “recycles” the hidden state across batches by detaching it from the computation graph each time.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import RNNLanguageModel
from dataProcessing import data_processing
import matplotlib.pyplot as plt


# Load processed data from your dataProcessing step.
############# add option to build the .pth if not already present.

data = torch.load("processed_data.pth", weights_only=False)
train_loader = data['train_loader']
valid_loader = data['valid_loader']
vocab_size = data.get('vocab_size', 10001)


# Hyperparameters.
embedding_dim = 100           
hidden_dim = 256              
dropout = 0.5                 
epochs = 20                    # Train for 20 epochs.
learning_rate = 0.001          # Learning rate for Adam optimizer.

# Determine device: GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model.
model = RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

def detach_hidden(hidden):
    # Detach hidden state to prevent backpropagating through the entire history.
    return hidden.detach()

print("Starting training...")
for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0.0
    hidden = None  # Will be initialized at the start of the first batch.
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = inputs.size(0)
        
        # Initialize hidden state if needed or adjust size.
        if hidden is None or hidden.size(1) != batch_size:
            hidden = model.init_hidden(batch_size, device)
        else:
            hidden = detach_hidden(hidden)
        
        optimizer.zero_grad()
        # Forward pass.
        outputs, hidden = model(inputs, hidden)
        # Reshape outputs and targets to compute loss.
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    perplexity = torch.exp(torch.tensor(avg_loss))
    train_losses.append(avg_loss)
    train_perplexities.append(perplexity.item())
    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f} - Perplexity: {perplexity:.2f}")
    
    # Evaluate on validation data.
    model.eval()
    valid_loss = 0.0
    hidden_valid = None
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            if hidden_valid is None or hidden_valid.size(1) != batch_size:
                hidden_valid = model.init_hidden(batch_size, device)
            else:
                hidden_valid = detach_hidden(hidden_valid)
            outputs, hidden_valid = model(inputs, hidden_valid)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            valid_loss += loss.item()
    avg_valid_loss = valid_loss / len(valid_loader)
    valid_perplexity = torch.exp(torch.tensor(avg_valid_loss))
    valid_losses.append(avg_valid_loss)
    valid_perplexities.append(valid_perplexity.item())
    print(f"Validation - Loss: {avg_valid_loss:.4f} - Perplexity: {valid_perplexity:.2f}")

# Save the trained model weights.
torch.save(model.state_dict(), "rnn_language_model.pth")
torch.save(model.state_dict(), 'weights/rnn_language_model.pth')

print("Model training complete and weights saved.")

# Plot loss and perplexity
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, epochs + 1), valid_losses, label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_perplexities, label='Train Perplexity')
plt.plot(range(1, epochs + 1), valid_perplexities, label='Valid Perplexity')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.title('Perplexity Over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('training_plot.png')
