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
import re
import numpy as np
import pandas as pd
import urllib.request
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


################################################################################
TWEET_LENGTH = 280

NUM_EPOCHS = 5
BUFFER_SIZE = 10000
BATCH_SIZE = 64
EMBEDDING_DIM = 256
NUM_RNN_UNITS = 2048
MAX_SEQ_LENGTH = 100
DROP_RATE = 0.5

GENERATED_DIR = os.path.join(os.getcwd(), "generated")
SAVE_DIR = os.path.join(os.getcwd(), "saved\\weights")
START_STRING = "Make America"
TEMPERATURE = 0.1
