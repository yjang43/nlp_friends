import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim)
        self.lin = nn.Linear(embedding_dim, num_embeddings)

    def forward(self, x):
        # there will be batch
        # x comes in a dim of (W, B)
        context_vectors = self.emb(x)
        context_vector = context_vectors.mean(0)    # take mean and shrink it to size (1, B, E)
        output = self.lin(context_vector)
        # drop out?
        return output
