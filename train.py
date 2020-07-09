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
