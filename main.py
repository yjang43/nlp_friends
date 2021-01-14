""" This file runs conversational bot in terminal.
"""

import torch
import torch.nn.functional as F
from nltk import word_tokenize

from model import DialogTransformer
from loader import FriendsDataset

# load data, model, functions

load = torch.load('result/train.pt', torch.device('cpu'))
transformer_state_dict = load['transformer_state_dict']
transformer_kwargs = load['transformer_kwargs']
emb = load['emb']
vocab = load['vocab']
txt_idx = load['txt_idx']
idx_txt = load['idx_txt']
dataset = load['dataset']
max_len = load['max_len']

model = DialogTransformer(**transformer_kwargs)
model.load_state_dict(transformer_state_dict)
model.eval()

def decode(request):
    request = word_tokenize(request)
    request = request + [dataset.BLANK_WORD] * (max_len - len(request))
    request = [txt_idx[t] if t in vocab else txt_idx[None] for t in request]
    response = greedy_decode(request)
    response = [idx_txt[i] for i in response]
    response = " ".join(response)
    return response

def greedy_decode(request):
    source = torch.tensor(request, dtype=torch.long).unsqueeze(1)
    target = torch.tensor([txt_idx[dataset.BOS_WORD]], dtype=torch.long).unsqueeze(1)
    response = []

    for i in range(1, max_len):
        output = model(source, target).view(i, -1)
        output = F.softmax(output, 1)
        _, pred = torch.topk(output, 2, 1)
        next_idx = pred[-1, 0] if pred[-1, 0] != txt_idx[None] else pred[-1, 1]
        if next_idx.item() == txt_idx[dataset.EOS_WORD]:
            break
        target = torch.cat((target, next_idx.unsqueeze(0).unsqueeze(0)), dim=0)
        response.append(next_idx.item())
    return response
        
        


if __name__ == "__main__":
    print("Welcome to Friends chat bot")
    print("***************")
    print("PRETRAIN SETTING")
    if 'pretrain_param' in load:
        print(load['pretrain_param'])
        print("Pretrain time:", load['pretrain_time'])
    print()
    print()
    print("TRAIN SETTING")
    if 'train_param' in load:
        print(load['train_param'])
        print("Train time:", load['train_time'])
    print("***************")
    
    try:
        while True:
            request = input("YOU: ")
            response = decode(request)
            print(f"COM: {response}")
    except KeyboardInterrupt as e:
        print("\nfinish")   # finish statement
        