import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
# from rnn import RNN
# import torch.optim as optim
from collections import Counter


def data_processing(): 

    ########## Step 1: Load the Text Data ##########

    train_file = "dataset/wiki2.train.txt"
    valid_file = "dataset/wiki2.valid.txt"
    test_file = "dataset/wiki2.test.txt"
    # Read the files.
    with open(train_file, "r", encoding="utf-8") as f:
        train_text = f.read()
    with open(valid_file, "r", encoding="utf-8") as f:
        valid_text = f.read()
    with open(test_file, "r", encoding="utf-8") as f:
        test_text = f.read()
    #Verify Data Type and Size
    # print("Train Data Type:", type(train_text))  # Should be <class 'str'>
    # print("Train Data Length:", len(train_text))  # Number of characters
    # print("Train Data Sample:", train_text[:500])  # Print first 500 characters
    # print("\nValid Data Length:", len(valid_text))
    # print("Test Data Length:", len(test_text))


    ########## Step 2: Tokenization (White Space Based) ##########

    train_tokens = train_text.split(" ")
    valid_tokens = valid_text.split(" ")
    test_tokens = test_text.split(" ")
    # Check the number of tokens
    print("Number of Train Tokens:", len(train_tokens))
    print("Number of Valid Tokens:", len(valid_tokens))
    print("Number of Test Tokens:", len(test_tokens))
    # print("Train Tokens Sample:", train_tokens[:20])


    ########## Step 3: Vocabulary Creation ##########  
    
    # Count word frequencies in training data
    word_counts = Counter(train_tokens)
    vocab_size = 10_000
    most_common_words = word_counts.most_common(vocab_size)

    # Create word-to-index mapping
    word_to_index = {word: idx for idx, (word, _) in enumerate(most_common_words, start=1)}

    # Add special token for unknown words
    word_to_index["<unk>"] = 0

    # Reverse mapping for index-to-word conversion (useful for decoding)
    index_to_word = {idx: word for word, idx in word_to_index.items()}

    print("Vocabulary Size:", len(word_to_index))  # Should be 10,001 including <unk>



    ########## Step 4: Convert Tokens to Indexes ##########

    # Function to convert tokenized text to indices
    def tokens_to_indices(tokens, word_to_index):
        return [word_to_index.get(word, word_to_index["<unk>"]) for word in tokens]

    # Convert train, validation, and test sets
    train_indices = tokens_to_indices(train_tokens, word_to_index)
    valid_indices = tokens_to_indices(valid_tokens, word_to_index)
    test_indices = tokens_to_indices(test_tokens, word_to_index)

    # Print sample of converted indices
    print("Train Indices Sample:", train_indices[:20])
    print("Valid Indices Sample:", valid_indices[:20])
    print("Test Indices Sample:", test_indices[:20])


    ########## Step 5: Convert Indexed Data into Sequences for Model Input ##########    

    #Create Input-Target Sequences
    seq_length = 30  # Number of tokens per input sequence
    batch_size = 64  # Adjust based on resources

    # Function to create input-target pairs
    def create_sequences(data, seq_length):
        inputs, targets = [], []
        for i in range(len(data) - seq_length):
            inputs.append(data[i:i+seq_length])
            targets.append(data[i+1:i+seq_length+1])  # Shifted by one position
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    # Generate sequences for train, validation, and test sets
    train_inputs, train_targets = create_sequences(train_indices, seq_length)
    valid_inputs, valid_targets = create_sequences(valid_indices, seq_length)
    test_inputs, test_targets = create_sequences(test_indices, seq_length)

    ########## Step 6: Wrap Sequences in DataLoader ##########

    # Create PyTorch DataLoaders for batch processing
    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_inputs, valid_targets), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_inputs, test_targets), batch_size=batch_size, shuffle=False)

    # Verify dataset shapes
    print("\nTrain Dataset Shape:", train_inputs.shape, train_targets.shape)
    print("Validation Dataset Shape:", valid_inputs.shape, valid_targets.shape)
    print("Test Dataset Shape:", test_inputs.shape, test_targets.shape)


    ########## Step 7: Save Processed Data ##########

    # Save processed data
    torch.save({
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'test_loader': test_loader
    }, "processed_data.pth")

    print("Processed data saved successfully!")



# Test execution loop
if __name__ == "__main__":
    data_processing()