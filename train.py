"""
Michael Patel
July 2020

Project description:
    Generate tweets in the style of President Trump

File description:
    For model preprocessing and training
"""
################################################################################
# Imports
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from model import *
from parameters import *


################################################################################
# chop tweets into <= 280 chunks
def chop(s, l):  # chop into 280 character long sequences
    if len(s) < TWEET_LENGTH + 1:
        l.append(s)
    else:
        a = s[:TWEET_LENGTH]
        b = s[TWEET_LENGTH:]
        chop(a, l)
        chop(b, l)


################################################################################
# Main
if __name__ == "__main__":
    print(f'TF version: {tf.__version__}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    # read in CSV data
    csv_filepath = os.path.join(os.getcwd(), "data\\tweets.csv")
    df = pd.read_csv(csv_filepath, encoding="windows-1252")

    tweets = df["Tweet_Text"]
    num_tweets = len(tweets)
    print(f'Number of tweets: {num_tweets}')

    # character-based, so no need for tf.keras.preprocessing.text.Tokenizer
    # convert to sequences of ints
    unique_chars = sorted(set("\n".join(tweets)))
    vocab_size = len(unique_chars)  # number of unique characters

    # create mapping from character > int
    char2int = {u: i for i, u in enumerate(unique_chars)}

    # create mapping from int > character
    int2char = {i: u for i, u in enumerate(unique_chars)}

    # convert to sequences of ints
    tweets_as_ints = []
    for tweet in tweets:
        line = [char2int[c] for c in tweet]
        chop(line, tweets_as_ints)

    # build n-gram sequences
    n_gram_sequences = []
    for t in tweets_as_ints:
        for i in range(1, len(t)):
            n_gram_sequences.append(t[:i+1])

    print(f'Number of n-gram sequences: {len(n_gram_sequences)}')

    # pad sequences
    max_sequence_len = max([len(seq) for seq in n_gram_sequences])
    print(f'Max sequence length: {max_sequence_len}')
    input_seqs = np.array(tf.keras.preprocessing.sequence.pad_sequences(
        sequences=n_gram_sequences,
        maxlen=max_sequence_len,
        padding="pre"
    ))
    print(f'Number of padded sequences: {len(input_seqs)}')

    # build (features, labels)
    # features = sequences except last token
    # labels = just last token
    features = input_seqs[:, :-1]
    labels = input_seqs[:, -1]
    print(f'Shape of features: {features.shape}')
    print(f'Shape of labels: {labels.shape}')

    # shuffle and batch
    sequences = tf.data.Dataset.from_tensor_slices((features, labels))
    sequences = sequences.shuffle(buffer_size=BUFFER_SIZE)
    sequences = sequences.batch(batch_size=BATCH_SIZE, drop_remainder=True)

    print(f'Shape of batch sequences: {sequences}')

    # ----- MODEL ----- #
    model = build_model(
        vocab_size=vocab_size,
        batch_size=BATCH_SIZE
    )

    model.summary()

    # ----- TRAIN ----- #
