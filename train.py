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
from data import Data
from model import *
from parameters import *


################################################################################
# loss function
def loss_fn(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true=labels,
        y_pred=logits,
        from_logits=True
    )


################################################################################
# Main
if __name__ == "__main__":
    print(f'TF version: {tf.__version__}')

    # ----- ETL ----- #
    # ETL = Extraction, Transformation, Load
    data = Data()
    tweets = data.get_tweet_string()
    vocab_size = data.get_vocab_size()

    # build (features, labels)
    features = []
    labels = []

    # build list of sequences of int tokens
    for i in range(0, len(tweets)-MAX_SEQ_LENGTH, MAX_SEQ_LENGTH):
        # create batch of char (i.e. list of char)
        f = tweets[i: i+MAX_SEQ_LENGTH]  # all char in chunk, except last
        l = tweets[i+1: i+1+MAX_SEQ_LENGTH]  # all char in chunk, except first

        # convert each char in chunk to int
        features.append([data.char2int[c] for c in f])
        labels.append([data.char2int[c] for c in l])

    print(f'Shape of features: {np.array(features).shape}')
    print(f'Shape of labels: {np.array(labels).shape}')

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
        loss=loss_fn,
        optimizer=tf.keras.optimizers.Adam()
    )

    # ----- TRAIN ----- #
    history = model.fit(
        #x=sequences.repeat(),
        x=sequences,
        epochs=NUM_EPOCHS
        #steps_per_epoch=len(tweets) // BATCH_SIZE // MAX_SEQ_LENGTH
    )

    # save weights > for generation, need to rebuild model with batch_size=1
    model.save_weights(SAVE_DIR)

    # plot of loss function
    #plt.style.use("dark_background")
    plt.figure(figsize=(20, 10))
    plt.scatter(range(1, NUM_EPOCHS+1), history.history["loss"], label="loss", s=500)
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(os.getcwd(), "results\\training_loss"))
