"""
Michael Patel
July 2020

Project description:
    Generate tweets in the style of President Trump

File description:
    For model inference
"""
################################################################################
# Imports
import os
import numpy as np
import tensorflow as tf

from data import Data
from model import build_model
from parameters import *


################################################################################
# generate text output
def generate_text(model, input_string):
    data = Data()

    # vectorize input_string
    input_seq = [data.char2int[c] for c in input_string]
    input_seq = tf.expand_dims(input_seq, 0)

    gen_text = []

    model.reset_states()

    num_gen_char = np.random.randint(140, 280)

    for i in range(num_gen_char):
        predictions = model(input_seq)
        predictions = tf.squeeze(predictions, 0)  # remove batch dimension

        predictions = predictions / TEMPERATURE

        # use categorigal distribution to predict the char returned by model
        id_predictions = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # pass predicted char as next input to model along with previous hidden state
        input_seq = tf.expand_dims([id_predictions], 0)

        gen_text.append(data.int2char[id_predictions])

    return input_string + "".join(gen_text)


################################################################################
# Main
if __name__ == "__main__":
    data = Data()

    # load weights
    model = build_model(
        vocab_size=data.get_vocab_size(),
        batch_size=1
    )

    weights_filepath = os.path.join(SAVE_DIR)
    model.load_weights(weights_filepath)
    model.build(tf.TensorShape([1, None]))
    model.summary()

    # generate an output sequence
    generated_message = generate_text(
        model=model,
        input_string=START_STRING
    )
    print(f'Generated message: {generated_message}')
