import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01 # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01 # Hidden to output

        # Initialize biases
        self.bh = np.zeros((hidden_size, 1)) # Hidden bias
        self.by = np.zeros((output_size, 1)) # Output bias

    def forward(self, inputs, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)

        # Forward pass
        for t in range(len(inputs)):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][inputs[t]] = 1  # One-hot encoding of input
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # Hidden state
            ys[t] = np.dot(self.Why, hs[t]) + self.by  # Unnormalized log probabilities for next chars
            ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))  # Probabilities for next chars
        return xs, hs, ps

    def sample(self, seed_ix, n):
        x = np.zeros((self.input_size, 1))
        x[seed_ix] = 1
        ixes = []
        h = np.zeros((self.hidden_size, 1))
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y) / np.sum(np.exp(y))
            ix = np.random.choice(range(self.output_size), p=p.ravel())
            x = np.zeros((self.input_size, 1))
            x[ix] = 1
            ixes.append(ix)
        return ixes

# Example usage
data = "hello world"
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

rnn = SimpleRNN(vocab_size, 100, vocab_size)

# Training parameters
learning_rate = 0.1

# Training
n, p = 0, 0
inputs = [char_to_ix[ch] for ch in data[:-1]]
targets = [char_to_ix[ch] for ch in data[1:]]

while True:
    if p + rnn.input_size + 1 >= len(data) or n == 0:
        hprev = np.zeros((rnn.hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p + rnn.input_size]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + rnn.input_size + 1]]

    xs, hs, ps = rnn.forward(inputs, hprev)

    # Loss computation
    loss = -np.sum([np.log(ps[t][targets[t], 0]) for t in range(len(inputs))])

    # Backpropagation
    dWxh, dWhh, dWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
    dbh, dby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += np.dot(dy, hs[t].T)
        dby += dy
        dh = np.dot(rnn.Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(rnn.Whh.T, dhraw)

    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)  # Clip gradients to mitigate exploding gradients

    # Update weights and biases
    rnn.Wxh -= learning_rate * dWxh
    rnn.Whh -= learning_rate * dWhh
    rnn.Why -= learning_rate * dWhy
    rnn.bh -= learning_rate * dbh
    rnn.by -= learning_rate * dby

    p += rnn.input_size
    n += 1

    if n % 100 == 0:
        sample_ix = rnn.sample(inputs[0], 20)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))
