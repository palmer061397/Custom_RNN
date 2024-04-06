import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, hprev):
        xs, hs, ys, ps = {}, {}, {}, {}
        hs[-1] = np.copy(hprev)

        for t in range(len(inputs)):
            xs[t] = np.zeros((self.input_size, 1))
            xs[t][inputs[t]] = 1
            hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)
            ys[t] = np.dot(self.Why, hs[t]) + self.by
            ps[t] = np.exp(ys[t] - np.max(ys[t])) / np.sum(np.exp(ys[t] - np.max(ys[t])))
        return xs, hs, ps

    def sample(self, seed_ix, n):
        x = np.zeros((self.input_size, 1))
        x[seed_ix] = 1
        ixes = []
        h = np.zeros((self.hidden_size, 1))
        for t in range(n):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
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
seq_length = 25
num_epochs = 1000

# Training
n, p = 0, 0
inputs = [char_to_ix[ch] for ch in data[:-1]]
targets = [char_to_ix[ch] for ch in data[1:]]
mWxh, mWhh, mWhy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
mbh, mby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
smooth_loss = -np.log(1.0/vocab_size) * seq_length

while n < num_epochs:
    if p + seq_length + 1 >= len(data):
        hprev = np.zeros((rnn.hidden_size, 1))
        p = 0

    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # Forward pass
    xs, hs, ps = rnn.forward(inputs, hprev)

    # Loss computation
    loss = -np.sum([np.log(ps[t][targets[t], 0]) for t in range(len(inputs))]) / seq_length

    # Backpropagation
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1
        dWhy = np.dot(dy, hs[t].T)
        dby = dy
        dh = np.dot(rnn.Why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh = dhraw
        dWxh = np.dot(dhraw, xs[t].T)
        dWhh = np.dot(dhraw, hs[t - 1].T)
        dhnext = np.dot(rnn.Whh.T, dhraw)

        # Accumulate gradients
        for param, dparam, mem in zip([rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by],
                                      [dWxh, dWhh, dWhy, dbh, dby],
                                      [mWxh, mWhh, mWhy, mbh, mby]):
            mem += dparam * dparam
            param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # Adagrad update

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    if n % 100 == 0:
        sample_ix = rnn.sample(inputs[0], 20)
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))
    p += seq_length
    n += 1
