import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

"""
Simple char prediction using one cell.
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cell = nn.RNN(input_size=4, hidden_size=4, batch_first=True)

    def forward(self, h, x):
        x = x.view(1, 1, 4)
        y, h = self.cell(x, h)

        return h, y.view(-1, 4)

    def init_hidden(self):
        return Variable(torch.zeros(1, 1, 4))


idx2char = ['h', 'e', 'l', 'o']

x_data = [0, 1, 2, 2]
one_hot_lookup = [[1, 0, 0, 0],  # 0
                  [0, 1, 0, 0],  # 1
                  [0, 0, 1, 0],  # 2
                  [0, 0, 0, 1]]  # 3

y_data = [1, 2, 2, 3]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = Variable(torch.Tensor(x_one_hot))
outputs = Variable(torch.LongTensor(y_data))

model = Model()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for e in range(100):
    optimizer.zero_grad()
    loss = 0
    h = model.init_hidden()

    for x, y in zip(inputs, outputs):
        h, y_pred = model(h, x)
        y = torch.LongTensor([y])
        loss += criterion(y_pred, y)

    print("e %3d, loss: %f" % (e + 1, loss))

    loss.backward()
    optimizer.step()


# output
with torch.no_grad():
    for x, y in zip(inputs, outputs):
        h, y_pred = model(h, x)
        print(f"For {idx2char[np.argmax(x)]}, predicted {idx2char[np.argmax(y_pred)]}")
