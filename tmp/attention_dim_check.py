import torch
import math
import torch.nn.functional as F 

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

mask = torch.tensor([[False, True], [False, False]])
mask = torch.tensor([[True]])
x = torch.tensor([[1, 1, 1], [1, 1, 1]], dtype=torch.float)
x = torch.tensor([[1, 1, 1]], dtype=torch.float)
attention(x, x, x, mask)

print(torch.randn(2, 2).masked_fill(mask, 10))