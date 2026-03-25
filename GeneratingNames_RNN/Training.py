
import torch
import torch.nn as nn
import time
import math

criterion = nn.NLLLoss()
learning_rate = 0.0005

def train(rnn,category_tensor, input_line_tensor, target_line_tensor,n_categories):
      target_line_tensor.unsqueeze_(-1)
      hidden = rnn.initHidden()

      rnn.zero_grad()
      loss = torch.Tensor([0])

      for i in range(input_line_tensor.size(0)):
           output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
           l=criterion(output, target_line_tensor[i])
           loss += l

      loss.backward()

      for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)
      
      return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)




