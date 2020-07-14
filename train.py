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

from data import Data
from model import *
from parameters import *


################################################################################
# Main
if __name__ == "__main__":
    print(f'TF version: {tf.__version__}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    data = Data()
    input_seqs = data.get_padded_sequences()
    vocab_size = data.get_vocab_size()
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

    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        optimizer=tf.keras.optimizers.Adam()
    )

    # ----- TRAIN ----- #
    history = model.fit(
        x=sequences.repeat(),
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(input_seqs) // BATCH_SIZE
    )

    # save model
    #model.save(os.path.join(os.getcwd(), "saved_model"))

    # save weights > for generation, need to rebuild model with batch_size=1
