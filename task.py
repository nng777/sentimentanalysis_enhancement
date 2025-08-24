"""

1.Swap LSTM for GRU:
1.1.Modify the original app.py code to replace the LSTM layer with a GRU (keras.layers.GRU) layer.
1.2.Train the model for the same number of epochs.
1.3.Create Report1.md: What is the final test accuracy? How does it compare to the LSTM model's performance?

2.Implement a Bidirectional LSTM:
2.1.Go back to the original LSTM model.
2.2.Wrap the LSTM layer with keras.layers.Bidirectional.
2.3.Train the model.
2.4.Create Report2.md: What is the new test accuracy? Why might processing the sequence in both directions be beneficial for sentiment analysis?

3.Build a Stacked LSTM Model:
3.1.Create a deeper model by stacking two LSTM layers.
3.2.Remember to set return_sequences=True on the first LSTM layer so it passes its full output sequence to the next layer.
3.3.Train this new stacked model.
3.4.Create Report3.md: Does the stacked model improve accuracy? What are the potential trade-offs of using a deeper model?
"""