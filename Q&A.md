
# Q&A for README_TOPIC.md

### Q1: What is the fundamental difference between a standard neural network and an RNN?

**A:** Standard networks (like MLPs) assume all inputs are independent. RNNs are designed for sequential data, using a **hidden state** to maintain a "memory" of previous inputs. This allows them to understand context and order in data like text or time series.

---

### Q2: What is the "vanishing gradient problem" and which network does it primarily affect?

**A:** The vanishing gradient problem occurs during the training of deep networks, where the gradients (signals for updating weights) shrink exponentially as they are propagated backward through time. For long sequences, these gradients can become so small that the network stops learning from earlier elements. This is a major limitation of **simple RNNs**.

---

### Q3: How do LSTMs solve the vanishing gradient problem?

**A:** LSTMs introduce a **cell state** and three **gates** (forget, input, output). This architecture acts as a controlled memory system. The gates regulate the flow of information, allowing the network to explicitly decide what to remember, what to forget, and what to output. This helps preserve important signals over long sequences, mitigating the vanishing gradient issue.

---

### Q4: In the Keras sentiment analysis example, what is the purpose of the `Embedding` layer?

**A:** The `Embedding` layer transforms the integer-coded words (word indices) into dense, fixed-size vectors. Instead of using arbitrary integers, this layer learns a meaningful, continuous representation for each word where similar words have similar vector representations. It's the first step in preparing text data for an LSTM.

---

### Q5: What is the main difference between an LSTM and a GRU?

**A:** A GRU (Gated Recurrent Unit) is a simplified version of an LSTM. It combines the forget and input gates into a single "update gate" and has fewer parameters. This makes it computationally more efficient and sometimes faster to train, while often delivering comparable performance to an LSTM.

---

### Q6: When would you use a Bidirectional LSTM?

**A:** You would use a Bidirectional LSTM when the context from **both past and future** elements in the sequence is important for making a prediction. For example, in sentiment analysis, the meaning of a word can be clarified by words that appear later in the sentence. A Bidirectional LSTM processes the sequence from start-to-end and end-to-start, providing a richer context to the model.

---

### Q7: What is the purpose of `return_sequences=True` when stacking LSTMs?

**A:** When you stack LSTM layers, the first LSTM layer needs to pass its entire sequence of outputs to the second LSTM layer, not just the final hidden state. Setting `return_sequences=True` ensures that the LSTM layer outputs its hidden state for every time step, providing the necessary sequential input for the subsequent layer.
