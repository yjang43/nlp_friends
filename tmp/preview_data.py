""" Get familiar with data I am dealing with
"""

import json
import os

with open("json/friends_season_01.json") as f:
    obj = json.load(f)
    obj_keys = obj.keys()
    print(obj_keys)     # season_id and episodes
    episode_0 = obj['episodes'][0]
    print(episode_0.keys())
    print(type(episode_0['scenes']))    # scenes in a list
    scene_0 = episode_0['scenes'][0]
    print(scene_0.keys())       
    utterance_0 = scene_0['utterances'][0]
    print(utterance_0.keys())       # "each utterance is a list of sentences where tokens are split."
    print(utterance_0['transcript'])    # see the difference
    print(utterance_0['tokens'])
    print("utterance 0:")
    print(utterance_0)
    utterance_1 = scene_0['utterances'][1]
    print("utterance 1:")
    print(utterance_1)

