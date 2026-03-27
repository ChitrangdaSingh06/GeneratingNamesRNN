import RNN
import NetwrokTraining
import Data
import Sampling

print('# categories:', Data.n_categories, Data.all_categories)
print(Data.unicodeToAscii("O'Néàl"))

rnn = RNN.RNN(Data.n_letters, 128, Data.n_letters)

n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every ``plot_every`` ``iters``

import time
start = time.time()

for iter in range(1, n_iters + 1):
    output, loss = NetwrokTraining.train(rnn,*NetwrokTraining.randomTrainingExample())
    total_loss += loss

    if iter % print_every == 0:
        print('%s (%d %d%%) %.4f' % (NetwrokTraining.timeSince(start), iter, iter / n_iters * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

        
#Plotting the Losses
import matplotlib.pyplot as plt

plt.figure()
plt.plot(all_losses)
plt.show()


Sampling.samples(rnn,'Russian', 'RUS')

Sampling.samples(rnn,'German', 'GER')

Sampling.samples(rnn,'Spanish', 'SPA')

Sampling.samples(rnn,'Chinese', 'CHI')
