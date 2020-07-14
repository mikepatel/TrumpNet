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
import tensorflow as tf

from data import Data
from parameters import *


################################################################################
# generate text output
def generate_text():
    data = Data()


################################################################################
# Main
if __name__ == "__main__":
    model_filepath = os.path.join(os.getcwd(), "saved_model")
    model = tf.keras.models.load_model(model_filepath)
