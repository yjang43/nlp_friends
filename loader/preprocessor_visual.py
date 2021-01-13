""" This module analyze for a proper number of vocabulary to be used in
the training model.
"""

from preprocessor import Preprocessor
import matplotlib.pyplot as plt
import numpy as np


THRESHOLD = 10
vocab_sizes = []

for t in range(0, THRESHOLD):
    p = Preprocessor(t)

    ### ADD DATA ACCORDINGLY ###
    p.add_season("json/friends_season_01.json")
    # p.add_episode("json/friends_season_01.json", 0)
    # p.add_season("json/friends_season_02.json")
    # p.add_season("json/friends_season_03.json")
    ############################

    vocab_size = len(p.make_vocabulary())
    print(t, vocab_size)
    vocab_sizes.append(vocab_size)

plt.plot([t for t in range(THRESHOLD)], vocab_sizes)
plt.show()

