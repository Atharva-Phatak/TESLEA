from wordfreq import zipf_frequency
import numpy as np
from nltk.corpus import stopwords
import nltk
import torch


def shift_to_score(shift, target_shift, right_slope=0.25):
    if shift <= target_shift:
        score = shift / (target_shift + 0.001)
    else:
        score = 1.0 - right_slope * (shift - target_shift) / (target_shift + 0.001)
    return np.clip(score, 0, 1.0)


class LexicalScore:
    def __init__(self, target_shift, word_change_ratio):
        self.target_shift = target_shift
        self.stopwords = set(stopwords.words("english"))
        self.word_change_ratio = (
            word_change_ratio  # Number of words that we expect to be swapped
        )

    def word_score_func(self, w):
        return zipf_frequency(w, "en", wordlist="large")

    def is_good_word(self, w):
        if w.lower() in self.stopwords:
            return False
        return True

    def vocab_shift_score(self, txt1, txt2, printing=False):
        words1 = nltk.tokenize.word_tokenize(txt1)
        words2 = nltk.tokenize.word_tokenize(txt2)
        words1 = set([w.lower() for w in words1 if self.is_good_word(w)])
        words2 = set([w.lower() for w in words2 if self.is_good_word(w)])

        removed_words = words1 - words2
        added_words = words2 - words1
        target_n_words = int(self.word_change_ratio * txt1.count(" "))

        vocab_shift = 0.0
        if target_n_words == 0:
            vocab_shift = 1.0  # You're not expected to have done any shifts yet
        elif len(removed_words) > 0 and len(added_words) > 0:
            # The idea of this is that we should consider only the K most complicated words removed
            # And by what top K most complicated they were replaced with.
            # The idea being that adding a bunch of simple words, or removing simple words doesn't matter beyond a certain point.

            added_words_zipfs = [
                {"w": w, "zipf": self.word_score_func(w)} for w in added_words
            ]
            removed_words_zipfs = [
                {"w": w, "zipf": self.word_score_func(w)} for w in removed_words
            ]
            added_words_zipfs = sorted(added_words_zipfs, key=lambda x: x["zipf"])
            removed_words_zipfs = sorted(removed_words_zipfs, key=lambda x: x["zipf"])[
                :target_n_words
            ]

            removed_avg_zipfs = np.mean(
                [x["zipf"] for x in removed_words_zipfs[:target_n_words]]
            )
            added_avg_zipfs = np.mean(
                [
                    x["zipf"]
                    for x in added_words_zipfs[
                        : min(target_n_words, len(removed_words_zipfs))
                    ]
                ]
            )

            vocab_shift = (
                (added_avg_zipfs - removed_avg_zipfs)
                * len(removed_words_zipfs)
                / target_n_words
            )

        return vocab_shift

    def score(self, sources, generateds, partial=False, printing=False, **kwargs):
        scores = []
        for source, generated in zip(sources, generateds):
            if partial:
                source = " ".join(source.split(" ")[: generated.count(" ")])

            vshift = self.vocab_shift_score(source, generated, printing=printing)
            score = shift_to_score(vshift, self.target_shift)

            scores.append(score)

        scores = torch.FloatTensor(scores)
        scores = (0.3 + torch.clamp(scores, 0.05, 1.0) * 0.7).tolist()

        return {
            "scores": scores,
        }
