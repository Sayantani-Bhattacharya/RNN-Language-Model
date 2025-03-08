import torch
import torch.nn as nn
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
    print("Train Tokens Sample:", train_tokens[:20])


    ########## Step 3: Vocabulary Creation ##########



















# Test execution loop
if __name__ == "__main__":
    data_processing()