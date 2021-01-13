from copy import deepcopy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from nltk import word_tokenize

from loader import FriendsDataset, Preprocessor, make_batch
from model import DialogTransformer

torch.manual_seed(0)

p = Preprocessor(threshold=1)
# p.add_episode("json/friends_season_01.json", 0)
p.add_season("json/friends_season_01.json")
vocab = p.make_vocabulary()
txt_idx, idx_txt = p.map_vocabulary_to_index(vocab)

vocab_size = len(vocab)
max_len = 32
batch_size = 4

print("************************")
print("vocab size:", vocab_size)
print("max len:", max_len)
print("batch size:", batch_size)
print("************************")

dataset = FriendsDataset('json/friends_season_01.json', max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=make_batch)


model = DialogTransformer(nn.Embedding(num_embeddings=vocab_size, embedding_dim=64))
# optimizer = torch.optim.SGD(model.parameters(), lr=1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(epochs=100):
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_loss = 0
        for i_batch, sampled_batch in enumerate(dataloader):
            src_batch, tgt_batch = deepcopy(sampled_batch)    # shape: B x M this needs to be transposed
            for i in range(len(src_batch)):
                # print(src_batch[i], tgt_batch[i])
                for j in range(max_len):
                    src_batch[i][j] = txt_idx[src_batch[i][j]] if src_batch[i][j] in vocab else txt_idx[None]
                    tgt_batch[i][j] = txt_idx[tgt_batch[i][j]] if tgt_batch[i][j] in vocab else txt_idx[None]
            src_batch = torch.LongTensor(src_batch).transpose(0, 1)
            tgt_batch = torch.LongTensor(tgt_batch).transpose(0, 1)
            # M x B x E
            tgt_batch_x = tgt_batch[:-1]
            tgt_batch_y = tgt_batch[1:]

            optimizer.zero_grad()

            output = model(src_batch, tgt_batch_x)
            output = output.view(-1, vocab_size)

            loss = criterion(output, tgt_batch_y.flatten())
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss
            # prediction test
            if epoch % 100 == 99 and i_batch == 0:
                model.eval()
                output = model(src_batch, tgt_batch)
                src_batch, tgt_batch = deepcopy(sampled_batch)    # shape: B x M this needs to be transposed
                print("input:")
                print(src_batch[0])
                print("target:")
                print(tgt_batch[0])
                output = output[:, 0, :].view(max_len, -1)
                _, pred = torch.max(output, 1)
                pred = pred.numpy()
                pred = np.vectorize(lambda x: idx_txt[x])(pred)
                print(pred)
                model.train()


        print(f"epoch: {epoch}\tloss: {round(total_loss.item() / len(src_batch), 5)}\ttime: {round(time.time() - epoch_start_time, 2)}")



def greedy_decode(src):
    src = np.array(src)
    print(src)
    src = np.vectorize(lambda x: txt_idx[x] if x in vocab else txt_idx[None])(src)
    print(src)
    src = torch.from_numpy(src).unsqueeze(1)
    tgt = torch.LongTensor([txt_idx[dataset.BOS_WORD]]).unsqueeze(1)
    response = []
    for i in range(1, max_len):
        # output = model(src, tgt).view(i, -1)
        output = model(src, tgt).view(i, -1)
        print("source:", src.flatten())
        print("target:", tgt.flatten())
        # print("output size:", output.size())
        output = F.softmax(output, 1)
        # print("prob", output)
        val, prob = torch.max(output, 1)
        next_idx = prob[-1].unsqueeze(0)
        # print(val)
        # print(next_idx)
        if next_idx == txt_idx[dataset.EOS_WORD]:
            break
        response.append(idx_txt[next_idx.item()])
        tgt = torch.cat((tgt, next_idx.unsqueeze(0)), dim=0)
        
    print(response)


def driver():
    print("lets play")
    try:
        while True:
            inp = input("You:")
            # inp = "all right joey, be nice. so, does he have a hump? a hump and a hairpiece?"
            inp = word_tokenize(inp)
            inp = inp + [dataset.BLANK_WORD] * (max_len - len(inp))

            greedy_decode(inp)

    except KeyboardInterrupt:
        print("bye")

do_train = False

if do_train:
    model.train()
    load = torch.load('word2vec.pt')   # load from pre-train
    vocab = load['vocab']
    txt_idx = load['txt_idx']
    idx_txt = load['idx_txt']
    vocab_size = len(vocab)

    max_len = 32
    train()
    save = {'state_dict': model.state_dict(), 'vocab': vocab, 'txt_idx': txt_idx, 'idx_txt': idx_txt, 'dataloader': dataloader}
    torch.save(save, 'model.pt')
else:
    load = torch.load('model.pt')
    load = torch.load('model_010721.pt')
    model.load_state_dict(load['state_dict'])
    vocab = load['vocab']
    txt_idx = load['txt_idx']
    idx_txt = load['idx_txt']
    model.eval()
    driver()


# '<bos>' 'wait' ',' 'does' 'he' 'eat' 'chalk' '?