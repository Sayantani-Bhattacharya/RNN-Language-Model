## Points to Note:

1. Extract the vocabulary only from the training set. Since the model learns from the training data, we should determine the most frequent words only from train.txt. The validation and test sets should not influence vocabulary selection.
2. Thus we must convert words to their corresponding integer indices (using the vocabulary from the training set). Replace any out-of-vocabulary (OOV) words in validation and test sets with <unk>.
