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
from datetime import datetime
import urllib.request
from time import sleep
import tensorflow as tf

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from data import Data
from model import build_model
from parameters import *


################################################################################
# get specified element
def get_element(id):
    elem = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, id))
    )

    return elem


# generate tweet image
def generate_tweet():
    print()

# generate text output
def generate_text(model, input_string):
    data = Data()

    # vectorize input_string
    input_seq = [data.char2int[c] for c in input_string]
    input_seq = tf.expand_dims(input_seq, 0)

    gen_text = []

    model.reset_states()

    num_gen_char = np.random.randint(40, 280)

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

    # chromedriver
    chromedriver_filepath = os.path.join(GENERATED_DIR, "chromedriver.exe")
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(
        chromedriver_filepath,
        options=chrome_options
    )

    # open url
    url = "https://www.tweetgen.com/create/tweet.html"
    driver.get(url=url)

    # Theme (light)

    # Profile Picture
    profile_pic_filepath = os.path.join(GENERATED_DIR, "trump_profile.jpg")
    profile_pic_upload = get_element(id="pfpInput")
    profile_pic_upload.send_keys(profile_pic_filepath)

    # Name
    name = "Donald J. Trump"
    name_element = get_element(id="nameInput")
    name_element.send_keys(name)

    # Username
    username = "realDonaldTrump"
    username_element = get_element(id="usernameInput")
    username_element.send_keys(username)

    # Verified User
    verify_checkbox = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div[1]/form/div[6]/div/label"))
    )
    verify_checkbox.click()

    # Tweet Content
    tweet_content = generated_message
    tweet_box = get_element(id="tweetTextInput")
    tweet_box.send_keys(tweet_content)

    # Image (skip)

    # Time
    time = datetime.now().strftime("%H:%M")
    time_element = get_element(id="time")
    time_element.send_keys(time)

    # Date
    # Day
    day = datetime.now().strftime("%d")
    day_element = get_element(id="dayInput")
    day_element.send_keys(day)

    # Month
    month = datetime.now().strftime("%m")
    month_element = get_element(id="monthInput")
    month_element.send_keys(month)

    # Year
    year = datetime.now().strftime("%Y")
    year_element = get_element(id="yearInput")
    year_element.send_keys(year)

    # Retweets
    randomize_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/div/div[1]/form/div[11]/div[2]/button"))
    )
    randomize_button.click()

    # Likes

    # Client (skip)

    # Generate Image and Download
    generate_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//*[@id="downloadButton"]'))
    )
    generate_button.click()

    # download generated image
    sleep(1)
    gen_image = get_element(id="imageOutput")
    src = gen_image.get_attribute("src")
    urllib.request.urlretrieve(src, os.path.join(GENERATED_DIR, "generated.png"))

    driver.close()
