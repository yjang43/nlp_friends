# Project Note

The project aims to create a conversation bot using transformer.

### Dataset
Conversational data from [Friends](https://github.com/emorynlp/character-mining)

How do deal with words not in token?
-> for the words not over threshold, put into None

If not covered by vocabulary, return None and feed into the model

### Model
Transformer

Using greedy decode for inference stage

http://nlp.seas.harvard.edu/2018/04/03/attention.html#label-smoothing

Word2Vec CBOW

### changes to be made

main.py

Need to make operation of FriednsDataset and Preprosessor consistent.


### requirements

pytorch, tensorboard, nltk - punkt to use tokenize_word, matplotlib to visualize word cutoff

