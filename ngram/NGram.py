import operator
import sys
from functools import reduce

import torch
import torch.nn.functional as F
from termcolor import colored
from torch import nn, optim
from torch.autograd import Variable

CONTEXT_SIZE = int(sys.argv[1])
EMBEDDING_DIM = int(sys.argv[2])

corpus = reduce(operator.add, [l.strip().split() for l in open("./corpus.txt", "rt").readlines()])

ngram = [
    (
        tuple(corpus[i:i + CONTEXT_SIZE]),  # prev n-1 words
        corpus[i + CONTEXT_SIZE]  # pred n'th word
    )
    for i in range(len(corpus) - CONTEXT_SIZE)
]

vocabulary = set(corpus)
word_to_idx = {word: i for i, word in enumerate(vocabulary)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}


class NGram(nn.Module):
    def __init__(self, vocab_size, context_size, n_dim):
        super(NGram, self).__init__()

        self.n_word = vocab_size
        self.embedding = nn.Embedding(self.n_word, n_dim)
        self.linear1 = nn.Linear(context_size * n_dim, 128)
        self.linear2 = nn.Linear(128, self.n_word)


    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        return F.log_softmax(out)


# --

ngram_model = NGram(vocab_size=len(vocabulary), context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM)
criterion = nn.NLLLoss()
optimizer = optim.SGD(ngram_model.parameters(), lr=1e-3)

# train
num_epochs = 100

for ei in range(1, num_epochs + 1):
    running_loss = 0

    for data in ngram:
        prev_words, next_word = data
        prev_words = Variable(torch.LongTensor([word_to_idx[i] for i in prev_words]))
        next_word = Variable(torch.LongTensor([word_to_idx[next_word]]))

        # forward
        pred_next_word = ngram_model(prev_words)
        loss = criterion(pred_next_word, next_word)
        running_loss += loss.item()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("[%3d/%3d] loss: %.8f" % (ei, num_epochs, running_loss / len(vocabulary)))

# test


acc = 0
for i in range(len(ngram)):
    prev_words, next_word = ngram[i]
    prev_words = Variable(torch.LongTensor([word_to_idx[i] for i in prev_words]))
    pred_next_word = ngram_model(prev_words)
    _, predict_label = torch.max(pred_next_word, 1)
    predict_word = idx_to_word[predict_label.item()]

    if predict_label == word_to_idx[next_word]:
        acc += 1
        print(colored('[%5d] real: [%s], predicted: [%s]' % (i + 1, next_word, predict_word), "green"))
    else:
        print(colored('[%5d] real: [%s], predicted: [%s]' % (i + 1, next_word, predict_word), "red"))

print("ACC = %.8f" % (1.0 * acc / len(ngram)))
