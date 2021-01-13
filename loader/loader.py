""" This module contains loader function that creates sequential data tokenized
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader

class FriendsDataset(Dataset):
    # TODO: Make it general to include all 10 seasons
    def __init__(self, fp, max_len, BOS_WORD='<bos>', EOS_WORD='<eos>', BLANK_WORD='<blank>'):
        self.max_len = max_len
        self.BOS_WORD = BOS_WORD
        self.EOS_WORD = EOS_WORD 
        self.BLANK_WORD = BLANK_WORD
        # TODO: Make this to load data from a season
        # self.data = self.load_episode(fp, 0) # loading first episode data only for now
        self.data = self.load_season(fp) # loading season

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_season(self, fp):
        with open(fp) as f:
            obj = json.load(f)
        N = len(obj['episodes'])
        srcs, tgts = [], []
        for n in range(N):
            d = self.episode_data(obj, n)
            src, tgt = d[0], d[1]
            srcs += src
            tgts += tgt
            
        data = list(zip(srcs, tgts))
        return data


    def load_episode(self, fp, n):
        with open(fp) as f:
            obj = json.load(f)
        srcs, tgts = self.episode_data(obj, n)
        data = list(zip(srcs, tgts))
        return data

    def episode_data(self, obj, n):
        srcs, tgts = [], []
        episode = obj['episodes'][n]
        scenes = episode['scenes']
        for scene in scenes:
            src, tgt = self.make_pairs(scene)
            srcs += src
            tgts += tgt
        return srcs, tgts

    def make_pairs(self, scene):
        if self.max_len < 20:
            raise ValueError("max_len is TOO SMALL! EOS_WORD will never occur")
        src, tgt = [], []
        utterances = scene['utterances']
        for i in range(len(utterances) - 1):
            src_tokens = utterances[i]['tokens']
            tgt_tokens = utterances[i + 1]['tokens']
            src_utterance, tgt_utterance = [], []

            len_counter = 0
            for sentence in src_tokens:
                len_counter += len(sentence)
                src_utterance += sentence
            src_utterance = [token.lower() for token in src_utterance]  # lower case only
            if len_counter <= self.max_len:
                src_utterance = src_utterance + [self.BLANK_WORD] * (self.max_len - len_counter)
            else:
                src_utterance = src_utterance + [self.BLANK_WORD] * (self.max_len - len_counter)
                src_utterance = src_utterance[: self.max_len]

            len_counter = 2     # including EOS WORD (ew) and BOS WORD (sw)
            for sentence in tgt_tokens:
                len_counter += len(sentence)
                tgt_utterance += sentence
            tgt_utterance = [token.lower() for token in tgt_utterance]  # lower case only
            if len_counter <= self.max_len:
                tgt_utterance = [self.BOS_WORD] + tgt_utterance + [self.EOS_WORD] + [self.BLANK_WORD] * (self.max_len - len_counter) 
            else:
                tgt_utterance = [self.BOS_WORD] + tgt_utterance + [self.EOS_WORD] + [self.BLANK_WORD] * (self.max_len - len_counter) 
                tgt_utterance = tgt_utterance[: self.max_len]

            src.append(src_utterance)
            tgt.append(tgt_utterance)
        return src, tgt


# def make_batch(data):
#     src_batch = [d[0] for d in data]
#     tgt_batch = [d[1] for d in data]
#     return src_batch, tgt_batch

class FriendsDataloader(DataLoader):
    """ This class wraps dataloader with an appropriate collate_fn.
    """
    def __init__(self, dataset, batch_size, txt_idx):
        self.txt_idx = txt_idx
        super().__init__(dataset, batch_size, shuffle=True, collate_fn=self.make_batch, drop_last=True)
    
    def make_batch(self, data):
        """ collate_fn that batchifies into tensor in a shape of M x B. """
        src_batch = [[self.txt_idx[t] if t in self.txt_idx else self.txt_idx[None] for t in d[0]] for d in data]
        tgt_batch = [[self.txt_idx[t] if t in self.txt_idx else self.txt_idx[None] for t in d[1]] for d in data]
        src_batch = torch.tensor(src_batch, dtype=torch.long).transpose(0, 1)
        tgt_batch = torch.tensor(tgt_batch, dtype=torch.long).transpose(0, 1)
        return src_batch, tgt_batch




if __name__ == '__main__':
    ### Test ###
    fp = 'json/friends_season_01.json'
    dataset = FriendsDataset(fp, 32)
    from preprocessor import Preprocessor
    p = Preprocessor(1)
    p.add_season(fp)
    vocab = p.make_vocabulary()
    txt_idx, _ = p.map_vocabulary_to_index(vocab)
    
    dataloader = FriendsDataloader(dataset, batch_size=2, shuffle=True, txt_idx=txt_idx)
    for i_batch, sampled_batch in enumerate(dataloader):
        print(i_batch)
        print(torch.cat(sampled_batch, 0))
        break
