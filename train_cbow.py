import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2
train_corpus = open("text8.txt", "r")
raw_text = train_corpus.read().split()

vocab = set(raw_text)
vocab_size = len(vocab)

print(" Vocabulary의 크기는 {}입니다.".format(vocab_size))

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []

for i in range(2, len(raw_text) -2 ):
    context = [raw_text[i-2], raw_text[i-1], raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

pass