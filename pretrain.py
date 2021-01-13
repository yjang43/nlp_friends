""" This module trains CBOW.
"""
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import CBOW
from loader import FriendsDataset, FriendsDataloader, Preprocessor


# forming vocabulary
p = Preprocessor(threshold=3)
p.add_season('json/friends_season_01.json')
vocab = p.make_vocabulary()
txt_idx, idx_txt = p.map_vocabulary_to_index(vocab)
meta_vocab = [idx_txt[i] for i in sorted(idx_txt)]

# hyperparam
epochs = 100
batch_size = 4
embedding_dim = 128
num_embeddings = len(vocab) 
max_len = 32    # words within a data sample, margin filled with BLANK_WORD
lr = 0.0005
window_size = 4
print_every = 1
param_string = (
    f"Epochs: {epochs}\n"
    f"Learning rate: {lr}\n"
    f"Batch size: {batch_size}\n"
    f"Embedding dim: {embedding_dim}\n"
    f"Vocab size: {num_embeddings}\n"
    f"Sentence length: {max_len}\n"
    f"Window size: {window_size}\n"
    f"Print every: {print_every}")
    
# define loader
dataset = FriendsDataset('json/friends_season_01.json', max_len)
dataloader = FriendsDataloader(dataset, batch_size=batch_size, txt_idx=txt_idx)

# train
model = CBOW(num_embeddings, embedding_dim)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_start_time = time.time()

        for itr, batch in enumerate(dataloader):
            avg_loss = 0
            big_batch = torch.cat(batch, 1)

            for s in range(max_len):
                left_pad = torch.zeros((max(window_size - s, 0), batch_size * 2), dtype=torch.long)
                left_pad = left_pad.masked_fill(left_pad == 0, txt_idx[dataset.BLANK_WORD])
                right_pad = torch.zeros((max(window_size + s - max_len + 1, 0), batch_size * 2), dtype=torch.long)
                right_pad = right_pad.masked_fill(right_pad == 0, txt_idx[dataset.BLANK_WORD])
                l_window = big_batch[max(s - window_size, 0): s, :]
                r_window = big_batch[s + 1: min(s + window_size + 1, max_len + 1), :]

                x = torch.cat((left_pad, l_window, r_window, right_pad), 0)
                y = big_batch[s, :]

                optimizer.zero_grad()
                y_hat = model(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()

                avg_loss += loss

            avg_loss = avg_loss / max_len
            epoch_loss += avg_loss

        epoch_loss = epoch_loss / (len(dataloader) // batch_size)
        if epoch % print_every == 0:
            print(f"epoch: {epoch}\tepoch_loss: {round(epoch_loss.item(), 5)}\t time: {round(time.time() - epoch_start_time, 2)}")
            tb_writer.add_scalar('Pretrain/Loss', epoch_loss, global_step=epoch)
    

if __name__ == "__main__":
    print("********************")
    # TODO: write print out statement
    print("SETTING")
    print(param_string)
    print("********************")
    train_time = time.time()
    tb_writer = SummaryWriter()
    try:
        train()
    except KeyboardInterrupt:
        pass
    train_time = time.time() - train_time
    print(f"total train time: {train_time}")

    torch_obj = {
        'emb': model.emb,
        'vocab': vocab,
        'txt_idx': txt_idx,
        'idx_txt': idx_txt,
        'train_time': train_time,
        'pretrain_param': param_string
    }
    torch.save(torch_obj, 'result/pretrain.pt')

    tb_writer.add_graph(model, torch.zeros((2 * window_size, 2 * batch_size), dtype=torch.long))
    tb_writer.add_embedding(model.emb.weight, meta_vocab)
    tb_writer.close()
