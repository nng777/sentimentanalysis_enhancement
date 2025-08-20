
# Homework: Enhancing the Sentiment Analysis Model

Based on the concepts in `README_TOPIC.md`, your task is to experiment with and enhance the provided sentiment analysis model.

### Your Tasks:

1.  **Swap LSTM for GRU:**
    *   Modify the original code to replace the `LSTM` layer with a `GRU` (`keras.layers.GRU`) layer.
    *   Train the model for the same number of epochs.
    *   **Report:** What is the final test accuracy? How does it compare to the LSTM model's performance?

2.  **Implement a Bidirectional LSTM:**
    *   Go back to the original `LSTM` model.
    *   Wrap the `LSTM` layer with `keras.layers.Bidirectional`.
    *   Train the model.
    *   **Report:** What is the new test accuracy? Why might processing the sequence in both directions be beneficial for sentiment analysis?

3.  **Build a Stacked LSTM Model:**
    *   Create a deeper model by stacking two `LSTM` layers.
    *   Remember to set `return_sequences=True` on the first `LSTM` layer so it passes its full output sequence to the next layer.
    *   Train this new stacked model.
    *   **Report:** Does the stacked model improve accuracy? What are the potential trade-offs of using a deeper model?

