"""
Michael Patel
July 2020

Project description:
    Generate tweets in the style of President Trump

File description:
    For model defintions
"""
################################################################################
# Imports
from parameters import *


################################################################################
# GRU
def build_model(vocab_size, batch_size):
    m = tf.keras.Sequential()

    # Embedding layer
    m.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        batch_size=batch_size
    ))
    """
    # GRU layer 1
    m.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        units=NUM_RNN_UNITS,
        return_sequences=True,  # return full sequence
        stateful=True
    )))
    
    # GRU layer 2
    m.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
        units=NUM_RNN_UNITS,
        return_sequences=True,  # return full sequence
        stateful=True
    )))
    """
    m.add(tf.keras.layers.GRU(
        units=NUM_RNN_UNITS,
        return_sequences=True,
        stateful=True
    ))

    # Output layer
    m.add(tf.keras.layers.Dense(
        units=vocab_size
    ))

    return m
