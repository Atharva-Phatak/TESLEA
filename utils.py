import json
import torch
from nltk import word_tokenize
from string import punctuation


def save_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f, indent=4)


def compute_mean(l, k):
    return sum([o[k] for o in l]) / len(l)


def create_weight_vector(fname, exclude_tokens, num_weights):

    # prepare weights vectors
    weights = []
    with open(fname) as f:
        for line in filter(lambda l: len(l) > 0, f.readlines()):
            index, weight = line.strip().split()

            if int(index) not in exclude_tokens:
                weights.append((int(index), float(weight)))

    weights = [w for w in weights if w[1] < 0]

    if num_weights > -1:
        weights = weights[:num_weights]

    # split ids and weights
    ids = [x[0] for x in weights]
    weights = torch.tensor([abs(x[1]) for x in weights])
    return ids, weights
