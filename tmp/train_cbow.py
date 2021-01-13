from copy import deepcopy
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.word2vec import CBOW
from loader import FriendsDataset, Preprocessor, make_batch
torch.manual_seed(0)
# # test
# model = CBOW(16, 4)
# x = torch.LongTensor([[1, 2, 3], [4, 5, 6]]) # input (M, B) -> (2, 3)
# output = model(x)
# output = output.unsqueeze(0)
# y = torch.LongTensor([1, 2]) # ground truth (M) -> (2)
# criterion = nn.CrossEntropyLoss()
# loss = criterion(output, y)

# loading dataset


p = Preprocessor(threshold=5)
# p.add_episode("json/friends_season_01.json", 0)
p.add_season("json/friends_season_01.json")
vocab = p.make_vocabulary()
txt_idx, idx_txt = p.map_vocabulary_to_index(vocab)

meta_vocab = [idx_txt[i] for i in sorted(idx_txt)]

num_embeddings = len(vocab)
embedding_dim = 128
max_len = 32

model = CBOW(num_embeddings, embedding_dim)

dataset = FriendsDataset('json/friends_season_01.json', max_len)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=make_batch)

def train(epochs=300):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    window_size = 2
    for epoch in range(epochs):
        current_time = time.time()
        epoch_loss = 0

        for batch_idx, sampled_batch in enumerate(dataloader):
            avg_loss = 0
            src_batch, tgt_batch = deepcopy(sampled_batch)
            for i in range(len(src_batch)):
                for j in range(max_len):
                    src_batch[i][j] = txt_idx[src_batch[i][j]] if src_batch[i][j] in vocab else txt_idx[None]
                    tgt_batch[i][j] = txt_idx[tgt_batch[i][j]] if tgt_batch[i][j] in vocab else txt_idx[None]
            batch = src_batch + tgt_batch   # shape (2B, M)
            # print("check batch if length of batch is two times batch")
            # print(len(batch))
            batch = torch.LongTensor(batch).transpose(0, 1) # shape (M, 2B)
            # print(batch.size())
            # sliding window
            for s in range(batch.size()[0]):
                left_pad = torch.zeros((max(window_size - s, 0), len(src_batch) * 2), dtype=torch.long)
                left_pad = left_pad.masked_fill(left_pad == 0, txt_idx[dataset.BLANK_WORD])
                right_pad = torch.zeros((max(window_size + s - batch.size()[0] + 1, 0), len(src_batch) * 2), dtype=torch.long)
                right_pad = right_pad.masked_fill(right_pad == 0, txt_idx[dataset.BLANK_WORD])
                # print("left right padding size")
                # print(left_pad.size(), right_pad.size())
                l_window = batch[max(s - window_size, 0): s, :]
                r_window = batch[s + 1: min(s + window_size + 1, batch.size()[0] + 1), :]

                # print("r_window size")
                # print(r_window.size())
                x = torch.cat((left_pad, l_window, r_window, right_pad), 0)
                # print("x size:")
                # print(x.size())
                # print(x)

                optimizer.zero_grad()

                y = batch[s, :]
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()

                optimizer.step()

                avg_loss += loss
            avg_loss = avg_loss / batch.size()[0] / batch.size()[1]     # iterating over sentence and num of batches included in loss
            # print(f"\taverage loss: {avg_loss.item()}")
            epoch_loss += avg_loss
        if epoch % 10 == 0:
            writer.add_embedding(model.emb.weight, meta_vocab, tag=f"epoch: {epoch}")

        print(f"epoch: {epoch}\tepoch_loss: {epoch_loss.item()}\t time: {round(time.time() - current_time, 2)}")   # this may be in accurate but still gives you an idea

if True:
    writer = SummaryWriter()
    train()
    torch.save({'state_dict': model.state_dict(),'txt_idx': txt_idx, 'idx_txt': idx_txt, 'dataset': dataset, 'vocab': vocab}, "word2vec.pt")
    writer.close()
else:
    pass