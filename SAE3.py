import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


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
lstm_units = 64

model = Sequential([
    # Layer 1: The Embedding Layer
    # This layer is like a sophisticated dictionary that turns our word-numbers (e.g., 34)
    # into meaningful vectors (e.g., [0.1, -0.4, 0.8, ...]).
    Embedding(input_dim=num_words, output_dim=embedding_dim, input_length=maxlen),

    # Layer 2: First LSTM Layer
    # This layer processes the sequence of word vectors and passes the full sequence
    # to the next LSTM layer by setting return_sequences=True.
    LSTM(units=lstm_units, return_sequences=True),

    # Layer 3: Second LSTM Layer
    # This layer receives the sequence from the previous LSTM and condenses it into
    # a single context vector representing the review.
    LSTM(units=lstm_units),

    # Layer 4: The Output Layer
    # This is a single neuron that will output one number between 0 and 1.
    # 0 will mean "negative review" and 1 will mean "positive review".
    Dense(1, activation='sigmoid')
])

model.summary()

# --- Simplified Explanation ---
# We're giving the model its instructions for how to learn.
model.compile(
    # 'adam' is an efficient algorithm for finding the best way to adjust the model's brain cells.
    optimizer='adam',
    # 'binary_crossentropy' is the math formula to calculate how wrong the model's predictions are.
    # It's used when the answer is one of two choices (positive/negative).
    loss='binary_crossentropy',
    # We want to measure the 'accuracy' (how many reviews it gets right).
    metrics=['accuracy']
)

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
    # We check its performance on the test data after each epoch to see how it's doing.
    validation_data=(X_test, y_test)
)

# --- Simplified Explanation ---
# Now that training is done, we do one final check on the test data
# to see how well our model learned to classify reviews it has never seen before.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")
