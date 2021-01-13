""" This module trains transformer.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import DialogTransformer
from loader import FriendsDataset, FriendsDataloader


# load pretrained data
load = torch.load('result/pretrain.pt')
emb = load['emb']
vocab = load['vocab']
txt_idx = load['txt_idx']
idx_txt = load['idx_txt']

# hyperparam
epochs = 100
batch_size = 4
max_len = 32    
lr = 0.0001
transformer_kwargs = {
    'embedding': emb,
    'nhead': 1,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'dim_feedforward': 1024,
}
print_every = 1
show_pred = False
param_string = (
    f"Epochs: {epochs}\n"
    f"Learning rate: {lr}\n"
    f"Batch size: {batch_size}\n"
    f"Transformer hyperparam: {transformer_kwargs}\n"
    f"Print every: {print_every}\n"
    f"Show prediction: {show_pred}")

# define loader
dataset = FriendsDataset('json/friends_season_01.json', max_len)
dataloader = FriendsDataloader(dataset, batch_size=batch_size, txt_idx=txt_idx)

# train
# # TODO: compare without pretrain
# transformer_kwargs['embedding'] = nn.Embedding(num_embeddings=emb.num_embeddings, embedding_dim=emb.embedding_dim)     
model = DialogTransformer(**transformer_kwargs)
optimizer = torch.optim.Adam(model.parameters(), lr)
criterion = nn.CrossEntropyLoss()

def train():
    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        for itr, batch in enumerate(dataloader):
            src_batch, tgt_batch = batch    # M x B

            tgt_batch_in = tgt_batch[:-1]   # target batch as an input
            tgt_batch_out = tgt_batch[1:]   # target batch as an output, note shift by one

            optimizer.zero_grad()
            output = model(src_batch, tgt_batch_in)

            if itr == 0 and show_pred: 
                # compare output to ground truth
                print(tgt_batch_out[:,0].size())
                sm_output = F.softmax(output[:, 0, :], 1)
                _, mx_pred = torch.max(sm_output, 1)
                print("INPUT")
                print([idx_txt[i.item()] for i in src_batch[:, 0]])
                print("PREDICTION")
                print([idx_txt[i.item()] for i in mx_pred])
                print("GROUND TRUTH")
                print([idx_txt[i.item()] for i in tgt_batch_out[:, 0]])

            output = output.view(-1, len(vocab))    # squeeze batch

            loss = criterion(output, tgt_batch_out.flatten())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss
        epoch_loss = epoch_loss / (len(dataloader) // batch_size)
        if epoch % print_every == 0:
            print(f"epoch: {epoch}\tepoch_loss: {round(epoch_loss.item(), 5)}\t time: {round(time.time() - epoch_start_time, 2)}")
            tb_writer.add_scalar('Train/Loss', epoch_loss, global_step=epoch)
    

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
        'transformer_state_dict': model.state_dict(),
        'transformer_kwargs': transformer_kwargs,
        'emb': emb,
        'vocab': vocab,
        'txt_idx': txt_idx,
        'idx_txt': idx_txt,
        'max_len': max_len,
        'batch_size': batch_size,
        'pretrain_time': load['train_time'],
        'train_time': train_time,
        'dataset': dataset,
        'train_param': param_string,
        'pretrain_param': load['pretrain_param']
    }
    torch.save(torch_obj, 'result/train.pt')

    tb_writer.add_graph(model, (torch.zeros((max_len, batch_size), dtype=torch.long), 
                        torch.zeros((max_len - 1, batch_size), dtype=torch.long)))
    tb_writer.close()