# Implement perplexity
# Implement cosine distance(semantic similarity rewards)
# Move rouge rewards here
# Lexical complexity award.
from datasets import load_metric, load_dataset
import nltk
from nltk.corpus import stopwords
import nltk
import numpy as np
import torch
from lexical_score import LexicalScore
import textstat
from scipy.spatial import distance

# import sent2vec
from nltk import word_tokenize
from string import punctuation

METRIC = load_metric("rouge.py")
LEXICAL_SCORE = LexicalScore(word_change_ratio=0.15, target_shift=0.4)
BIO_SENT = sent2vec.Sent2vecModel()
BIO_SENT.load_model("biomed/BioSentVec_PubMed_MIMICIII-bigram_d700.bin")
STOPWORDS = set(stopwords.words("english"))


def preprocess_sentence(text):
    text = text.replace("/", " / ")
    text = text.replace(".-", " .- ")
    text = text.replace(".", " . ")
    text = text.replace("'", " ' ")
    text = text.lower()

    tokens = [
        token
        for token in word_tokenize(text)
        if token not in punctuation and token not in STOPWORDS
    ]

    return " ".join(tokens)


def calculate_rouge_reward(gen, src, tokenizer, decode, return_sents=False):
    # op = METRIC.compute(gen, src)

    if decode:
        decoded_preds = tokenizer.batch_decode(gen, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        src = src.cpu().numpy()
        labels = np.where(src != -100, src, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]
    else:
        decoded_preds = gen
        decoded_labels = src
    result = METRIC.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    metric = result["rouge2"]
    if return_sents:
        return {"generated": decoded_preds, "rouge": metric}
    else:
        return metric


def lexical_reward(gen, src):

    score = LEXICAL_SCORE.score(sources=src, generateds=gen)
    return score["scores"][0]


def fkgl_reward(gen, tgt):
    fkgl_gen = textstat.flesch_kincaid_grade(gen[0])
    fkgl_tgt = textstat.flesch_kincaid_grade(tgt[0])
    score = (fkgl_tgt - fkgl_gen) / (fkgl_tgt)
    return torch.tensor(score)


def cosine_similarity_reward(gen, tgt):
    gen = preprocess_sentence(gen[0])
    tgt = preprocess_sentence(tgt[0])
    gen_embed = BIO_SENT.embed_sentence(gen)
    src_embed = BIO_SENT.embed_sentence(tgt)
    cosine = 1 - distance.cosine(src_embed, gen_embed)
    # cosine = distance.cosine(src_embed, gen_embed)
    return torch.tensor(cosine)


def combine_rewards(gen, tgt, tokenizer, reward_list, decode=True, src=None):
    if decode:
        gen = tokenizer.batch_decode(
            gen, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        tgt = tokenizer.batch_decode(
            tgt, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        if src is not None:
            src = tokenizer.batch_decode(
                src, clean_up_tokenization_spaces=True, skip_special_tokens=True
            )

        total_rewards = {}
        for r in reward_list:
            r_name = r["func"].__name__.split("_reward")[0]
            if "lexical" in r_name:
                if src is None:
                    raise ValueError(
                        "Source sentence must be given if you want to compute lexical rewards."
                    )
                else:
                    total_rewards[r_name] = r["w"] * r["func"](gen=gen, src=src)
            else:
                total_rewards[r_name] = r["w"] * r["func"](gen=gen, tgt=tgt)
        return total_rewards
