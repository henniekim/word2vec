import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time

torch.manual_seed(1)

def main():
  EMBEDDING_DIM = 100
  if torch.cuda.is_available():
    VOCAB_SIZE = 30000
    print("cuda available")
  else:
    VOCAB_SIZE = 30000

  UNK_TOKEN = "<UNK>"
  WINDOW_SIZE = 5
  BATCH_SIZE = 10240

  words = []
  with open("data/text8.txt") as f:
    for line in f.readlines():
      words += line.strip().split(" ")

  print("total words in corpus: %d" % (len(words)))

  word_cnt = Counter()
  for w in words:
    if w not in word_cnt:
      word_cnt[w] = 0
    word_cnt[w] += 1

  # calculate word coverage of 30k most common words
  total = 0
  for cnt_tup in word_cnt.most_common(VOCAB_SIZE):
    total += cnt_tup[1]
  print("coverage: %.4f " % (total * 1.0 / len(words)))
  # 95.94%

  # make vocabulary with most common words
  word_to_ix = dict()
  for i, cnt_tup in enumerate(word_cnt.most_common(VOCAB_SIZE)):
    word_to_ix[cnt_tup[0]] = i

  # add unk token to vocabulary
  word_to_ix[UNK_TOKEN] = len(word_to_ix)

  # replace rare words in train data with UNK_TOKEN
  train_words = []
  for w in words:
    if w not in word_to_ix:
      train_words += [UNK_TOKEN]
    else:
      train_words += [w]

  word_to_ix_list = []

  for key, value in word_to_ix.items():
      temp = [key,value]
      word_to_ix_list.append(temp)

  # make train samples for SkipGram
  train_input = []
  train_target = []
  span = (WINDOW_SIZE - 1) // 2

  print(len(train_words))

  for i in range(span, int(len(train_words)) - span):
    context = []
    for j in range(WINDOW_SIZE):
      if j != span:
        context.append(word_to_ix[train_words[i + j - span]])
    target = word_to_ix[train_words[i]]
    train_target.append(context)
    train_input.append(target)
  print("data is generated!")
  pass


  # model class
  class SkipGram(nn.Module):

    def __init__(self, vocab_size, embedding_dim, window_size):
      super(SkipGram, self).__init__()
      self.embeddings = nn.Embedding(vocab_size, embedding_dim)
      self.linear1 = nn.Linear(embedding_dim, vocab_size)
      pass

    def forward(self, inputs):
      embeds = self.embeddings(inputs)
      out = self.linear1(torch.mean(embeds, dim=1))
      log_probs = F.log_softmax(out, dim=1)
      return log_probs

    def get_word_emdedding(self, word):
      word = torch.LongTensor([word_to_ix[word]])
      return self.embeddings(word).view(1, -1)

  # set up to train
  losses = []
  loss_function = nn.NLLLoss()
  model = SkipGram(len(word_to_ix), EMBEDDING_DIM, WINDOW_SIZE)
  # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer = optim.Adam(model.parameters(), lr=0.1)
  #
  # if gpu is available, then use it
  if torch.cuda.is_available():
    model.cuda()
    pass

  # make data loader for batch training
  train_input = torch.from_numpy(np.asarray(train_input)).long()
  train_target = torch.from_numpy(np.asarray(train_target)).long()
  dataset_train = data_utils.TensorDataset(train_input, train_target)
  train_loader = data_utils.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)

  # training loop
  for epoch in range(100):
    total_loss = 0
    start = time.time()
    for i, (target, context) in enumerate(train_loader):

      if torch.cuda.is_available():
        target = target.cuda()
        context = context.cuda()
        pass

      model.zero_grad()

      log_probs = model(context)

      loss = loss_function(log_probs, target)

      loss.backward()
      optimizer.step()

      total_loss += loss.item()

      if (i+1) % 10 == 0:
        print("loss: %.4f, steps: %dk" % (loss.item(), ((i+1)/10)))

    end = time.time()
    print("epochs: %d" % (epoch+1))
    print("time eplased: %d seconds" % (end-start))
    print("mean loss: %.4f" % (total_loss / (train_input.shape[0] // BATCH_SIZE)))
    torch.save(model.state_dict(), "model/skip.epoch{}.model".format(epoch))

  # Here you need to save the model's hidden layer which is V * D word embedding matrix.
  # Then, use the word embedding matrix to get vectors for word
if __name__ == '__main__':
  main()