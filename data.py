"""
Michael Patel
July 2020

Project description:
    Generate tweets in the style of President Trump

File description:
    For data gathering
"""
################################################################################
# Imports
import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from parameters import *


################################################################################
class Data:
    def __init__(self):
        # read in CSV data
        csv_filepath = os.path.join(os.getcwd(), "data\\tweets.csv")
        df = pd.read_csv(csv_filepath, encoding="windows-1252")

        tweets = df["Tweet_Text"]
        num_tweets = len(tweets)
        print(f'Number of tweets: {num_tweets}')

        # character-based, so no need for tf.keras.preprocessing.text.Tokenizer
        # convert to sequences of ints
        self.tweets_str = "\n".join(tweets)

        # remove obscure characters
        self.tweets_str = re.sub(
            "[^A-Za-z0-9 \n.!?'()@#$%&-]+",
            "",
            self.tweets_str
        )

        # unique characters in string
        unique_chars = sorted(set(self.tweets_str))

        # number of unique characters = vocab size
        self.vocab_size = len(unique_chars)  # number of unique characters
        print(f'Vocab size: {self.vocab_size}')

        # create mapping from character > int
        self.char2int = {u: i for i, u in enumerate(unique_chars)}

        # create mapping from int > character
        self.int2char = {i: u for i, u in enumerate(unique_chars)}

        """
        # convert to sequences of ints
        tweets_as_ints = []
        for tweet in tweets:
            line = [self.char2int[c] for c in tweet]
            self.chop(line, tweets_as_ints)
            #tweets_as_ints.append(line)

        # build n-gram sequences
        n_gram_sequences = []
        for t in tweets_as_ints:
            for i in range(1, len(t)):
                n_gram_sequences.append(t[:i+1])

        print(f'Number of n-gram sequences: {len(n_gram_sequences)}')
        #n_gram_sequences = tweets_as_ints

        # pad sequences
        max_sequence_len = max([len(seq) for seq in n_gram_sequences])
        print(f'Max sequence length: {max_sequence_len}')
        self.padded_sequences = np.array(tf.keras.preprocessing.sequence.pad_sequences(
            sequences=n_gram_sequences,
            maxlen=max_sequence_len,
            padding="pre"
        ))
        """

    # helper function to chop tweets into 280 size chunks
    def chop(self, in_line, out_list):
        if len(in_line) < TWEET_LENGTH + 1:
            out_list.append(in_line)
        else:
            a = in_line[:TWEET_LENGTH]
            b = in_line[TWEET_LENGTH:]
            self.chop(a, out_list)
            self.chop(b, out_list)

    # return padded sequences
    def get_padded_sequences(self):
        return self.padded_sequences

    # return tweets as string
    def get_tweet_string(self):
        return self.tweets_str

    # return vocab size
    def get_vocab_size(self):
        return self.vocab_size
