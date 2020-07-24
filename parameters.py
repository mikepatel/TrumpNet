"""
Michael Patel
July 2020

Project description:
    Generate tweets in the style of President Trump

File description:
    For model and training parameters
"""
################################################################################
# Imports
import os


################################################################################
TWEET_LENGTH = 280

NUM_EPOCHS = 6
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 2048
MAX_SEQ_LENGTH = 100

GENERATED_DIR = os.path.join(os.getcwd(), "generated")
SAVE_DIR = os.path.join(os.getcwd(), "saved\\weights")
START_STRING = "Make America"
TEMPERATURE = 0.3
