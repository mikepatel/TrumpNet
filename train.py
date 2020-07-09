"""
Michael Patel
July 2020

Project description:

File description:
"""
################################################################################
# Imports
import os
import pandas as pd
import tensorflow as tf


################################################################################
# Main
if __name__ == "__main__":
    print(f'TF version: {tf.__version__}')

    # read in CSV data
    csv_filepath = os.path.join(os.getcwd(), "data\\tweets.csv")
    df = pd.read_csv(csv_filepath, encoding="windows-1252")

    #
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
        tweets_as_ints.append(line)
    
    quit()

    # build n-gram sequences
    n_gram_sequences = []
    for t in tweets:
        for i in range(1, len(t)):
            n_gram_sequences.append(t[:i+1])

        #print(n_gram_sequences)

    # pad sequences
    quit()

    #tweets_str = "\n".join(tweets)
    #print(f'Length of tweet string: {len(tweets_str)}')
