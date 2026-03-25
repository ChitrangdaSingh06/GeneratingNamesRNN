import RNN
import time
import Training
import LoadData
import Sampling

rnn = RNN.RNN(LoadData.n_letters,128,LoadData.n_letters,LoadData.n_categories)
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 
start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = Training.train(rnn,*LoadData.randomTrainingExample(),n_categories=LoadData.n_categories)
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (Training.timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()

Sampling.samples(rnn,'Russian', 'RUS')

Sampling.samples(rnn,'German', 'GER')

Sampling.samples(rnn,'Spanish', 'SPA')

Sampling.samples(rnn,'Chinese', 'CHI')