""" This module contains a class that processes tokens.
Only tokens above a threshold is saved.
Dictionaries to convert token to index and vice versa is saved as attributes.
"""
import collections
import json


class Preprocessor:
    """ This class helps change text into usuable form for ML model.
    The class collects vocabulary from Friends transcript.
    """
    def __init__(self, threshold=1):
        self.threshold = threshold

        self.token_occurrence = collections.defaultdict(int)
    
    def add_season(self, fp):
        with open(fp) as f:
            obj = json.load(f)
        N = len(obj['episodes'])
        for n in range(N):
            self._add_episode(obj, n)

    def add_episode(self, fp, n):
        with open(fp) as f:
            obj = json.load(f)
        self._add_episode(obj, n)
    
    def _add_episode(self, obj, n):
        episode = obj['episodes'][n]
        scenes = episode['scenes']
        for scene in scenes:
            for utterance in scene['utterances']:
                for sentence in utterance['tokens']:
                    for token in sentence:
                        token = token.lower()   # lower case only
                        self.token_occurrence[token] += 1

    def make_vocabulary(self, BOS_WORD='<bos>', EOS_WORD='<eos>', BLANK_WORD='<blank>'):
        vocabulary = {token for token, occurrence in self.token_occurrence.items() if occurrence >= self.threshold}
        vocabulary.add(BOS_WORD)
        vocabulary.add(EOS_WORD)
        vocabulary.add(BLANK_WORD)
        vocabulary.add(None)        # outside vocabulary
        return vocabulary

    def map_vocabulary_to_index(self, vocabulary):
        # map text to index and vice versa
        txt_idx = {txt: idx for idx, txt in enumerate(vocabulary)}
        idx_txt = {idx: txt for idx, txt in enumerate(vocabulary)}
        return txt_idx, idx_txt