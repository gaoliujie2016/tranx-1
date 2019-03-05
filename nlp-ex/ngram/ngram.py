import operator
import sys
from functools import reduce

import torch
import torch.nn.functional as F
from termcolor import colored
from torch import nn, optim
from torch.autograd import Variable

file = sys.argv[1]
CONTEXT_SIZE = int(sys.argv[2])
EMBEDDING_DIM = int(sys.argv[3])
num_epochs = int(sys.argv[4])

to_lower = lambda s: list(map(lambda i: i.lower(), s))
strip_pct = lambda s: "".join(list(filter(str.isalpha, s)))

# read and normalize corpus
corpus = reduce(operator.add, [
    list(map(strip_pct, map(to_lower, l.strip().split())))
    for l in open(file, "rt").readlines()
])

ngrams = [
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
    def __init__(self, vocab_size, context_size, emb_dim):
        super(NGram, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, emb_dim)
        self.linear1 = nn.Linear(context_size * emb_dim, 128)
        self.linear2 = nn.Linear(128, self.vocab_size)


    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.view(1, -1)
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        return F.log_softmax(out)


# --

ngram_model = NGram(vocab_size=len(vocabulary), context_size=CONTEXT_SIZE, emb_dim=EMBEDDING_DIM)

criterion = nn.NLLLoss()
optimizer = optim.SGD(ngram_model.parameters(), lr=1e-3)

# train


for ei in range(1, num_epochs + 1):
    running_loss = 0

    for data in ngrams:
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
for i in range(len(ngrams)):
    prev_words, next_word = ngrams[i]
    emb_prev_words = Variable(torch.LongTensor([word_to_idx[i] for i in prev_words]))
    pred_next_word = ngram_model(emb_prev_words)
    _, pred_idx = torch.max(pred_next_word, 1)
    pred_word = idx_to_word[pred_idx.item()]

    if pred_idx == word_to_idx[next_word]:
        acc += 1
        print(colored('[%5d] real: [%s], predicted: [%s]' % (i + 1, next_word, pred_word), "green"))
    else:
        print(colored('[%5d] real: [%s], predicted: [%s]' % (i + 1, next_word, pred_word), "red"))

print("\nACC = %.8f\n" % (1.0 * acc / len(ngrams)))

# ---

while True:
    t = input("> ")
    t = Variable(torch.LongTensor([word_to_idx[i] for i in t.split()]))
    pred_next_word = ngram_model(t)
    _, pred_idx = torch.max(pred_next_word, 1)
    pred_word = idx_to_word[pred_idx.item()]
    print("predicted: [%s]" % pred_word)
