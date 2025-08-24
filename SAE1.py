import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense




def build_model(num_words: int, maxlen: int, embedding_dim: int, gru_units: int) -> Sequential:
    """Create and compile the sentiment analysis model."""
    model = Sequential([
        Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),
        GRU(units=gru_units),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    # --- Simplified Explanation ---
    # We are loading a famous dataset of movie reviews.
    # We only care about the 10,000 most common words to keep things manageable.
    num_words = 10000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    # Each review is just a list of numbers, where each number corresponds to a word.
    print("First review (raw integers):", X_train[0])

    # --- Simplified Explanation ---
    # Neural networks need inputs to be the same size.
    # Since reviews have different lengths, we'll make them all 200 words long.
    # If a review is shorter, we'll add blank spaces (padding). If it's longer, we'll cut it off.
    maxlen = 200

    # This function handles the padding for us.
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    print("First review (padded):", X_train[0])
    print("Padded training shape:", X_train.shape)

    # --- Simplified Explanation ---
    # We're defining the "brain" of our model layer by layer.
    embedding_dim = 128
    gru_units = 64

    model = build_model(num_words, maxlen, embedding_dim, gru_units)
    model.summary()

    # --- Simplified Explanation ---
    # This is where the learning happens!
    # The model will look at the training reviews and their labels (X_train, y_train)
    # over and over again (5 times, or 5 "epochs") to get better at predicting.
    batch_size = 64
    epochs = 5

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
    )

    # --- Simplified Explanation ---
    # Now that training is done, we do one final check on the test data
    # to see how well our model learned to classify reviews it has never seen before.
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")


if __name__ == "__main__":
    main()
