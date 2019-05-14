import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time

def main():
    torch.manual_seed(1)

    EMBEDDING_DIM = 100
    if torch.cuda.is_available():
        VOCAB_SIZE = 30000
    else:
        VOCAB_SIZE = 5000

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

    # make train samples for CBOW
    train_input = []
    train_target = []
    span = (WINDOW_SIZE - 1) // 2
    for i in range(span, len(train_words) - span):
        context = []
        for j in range(WINDOW_SIZE):
            if j != span:
                context.append(word_to_ix[train_words[i + j - span]])
        target = word_to_ix[train_words[i]]
        train_input.append(context)
        train_target.append(target)
    print("data is generated!")

    # model class
    class CBOW(nn.Module):

        def __init__(self, vocab_size, embedding_dim, window_size):
            super(CBOW, self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear1 = nn.Linear(embedding_dim, vocab_size)
            pass

        def forward(self, inputs):
            embeds = self.embeddings(inputs)
            out = self.linear1(torch.mean(embeds, dim=1))
            log_probs = F.log_softmax(out, dim=1)
            return log_probs

        def get_word_embedding(self, word):
            word = torch.LongTensor([word_to_ix[word]]).cuda()
            return self.embeddings(word).view(1, -1)

    # set up to train
    losses = []
    loss_function = nn.NLLLoss()
    model = CBOW(len(word_to_ix), EMBEDDING_DIM, WINDOW_SIZE)

    optimizer = optim.Adam(model.parameters(), lr=0.5)
    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # if gpu is available, then use it
    if torch.cuda.is_available():
        model.cuda()

    # make data loader for batch training
    train_input = torch.from_numpy(np.asarray(train_input)).long()
    train_target = torch.from_numpy(np.asarray(train_target)).long()
    dataset_train = data_utils.TensorDataset(train_input, train_target)
    train_loader = data_utils.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=False)

    # 편하게 target vector를 얻기 위해 함수 정의
    def word2vec(word):
        return torch.reshape(data[word_to_ix[word]], (1, EMBEDDING_DIM))

    wordList = []

    for key, value in word_to_ix.items():
        wordList.append([key, value])

    def arg2word(arg):
        return wordList[arg]

    # training loop
    for epoch in range(100):
        total_loss = 0
        start = time.time()
        for i, (context, target) in enumerate(train_loader):

            if torch.cuda.is_available():
                context = context.cuda()
                target = target.cuda()

            model.zero_grad()

            log_probs = model(context)

            loss = loss_function(log_probs, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i+1) % 10 == 0:
                print("loss: %.4f, steps: %dk" % (loss.item(), ((i+1)/10)))
                torch.save(model.state_dict(), "cbow.epoch{}.model".format(epoch))
                model_ = torch.load('cbow.epoch{}.model'.format(epoch))
                data = model_['embeddings.weight']

                # 일단 예제로
                athen_vec = model.get_word_embedding('athens')
                greece_vec = model.get_word_embedding('greece')
                baghdad_vec = model.get_word_embedding('baghdad')
                iraq_vec = model.get_word_embedding('iraq')

                result = torch.add(torch.sub(athen_vec, greece_vec), baghdad_vec)

                # athen = word2vec('athens')
                # greece = word2vec('greece')
                # baghdad = word2vec('baghdad')
                # iraq = word2vec('iraq')
                #
                # result_vec = athen - greece + baghdad
                #
                #
                # result = athen_vec - greece_vec + baghdad_vec
                result = torch.reshape(result, (1, EMBEDDING_DIM))

                # .cuda() 붙여줘야 GPU 연산 가능
                zeros = torch.tensor((), dtype=torch.float32).cuda()
                zeros = zeros.new_zeros((VOCAB_SIZE + 1, EMBEDDING_DIM))
                broadcasted_result = torch.add(zeros, result)

                norm_target = torch.nn.functional.normalize(broadcasted_result)
                norm_data = torch.nn.functional.normalize(data)

                cosine_similarity_result = torch.nn.functional.cosine_similarity(norm_target, norm_data)
                argmax2 = torch.argmax(cosine_similarity_result)

                # cosine_similarity_result = torch.nn.functional.cosine_similarity(broadcasted_result, data)

                # pairwise_distance = torch.nn.PairwiseDistance(keepdim=True)
                # pairwise_distance_result = pairwise_distance(broadcasted_result, data)
                # argmax = torch.argmax(pairwise_distance_result)

                argword2 = arg2word(argmax2)
                # gt = gt.lower()

                # argword = arg2word(argmax)



                # argmax = torch.argmax(cosine_similarity_result)
                # argmax = argmax.item()

                # print(wordList[argmax])

                print('sim to athens : '+ str(torch.nn.functional.cosine_similarity(norm_target, torch.nn.functional.normalize(model.get_word_embedding('athens')))))
                print('sim to greece : '+ str(torch.nn.functional.cosine_similarity(norm_target, torch.nn.functional.normalize(model.get_word_embedding('greece')))))
                print('sim to baghdad : '+ str(torch.nn.functional.cosine_similarity(norm_target, torch.nn.functional.normalize(model.get_word_embedding('baghdad')))))
                print('sim to iraq : '+ str(torch.nn.functional.cosine_similarity(norm_target, torch.nn.functional.normalize(model.get_word_embedding('iraq')))))

                print('sim to france : '+str(torch.nn.functional.cosine_similarity(norm_target, torch.nn.functional.normalize(model.get_word_embedding('france')))))

                # if wordList[argmax2][0] == 'longer':
                #     import sys
                #     sys.exit()

        end = time.time()
        print("epochs: %d" % (epoch+1))
        print("time eplased: %d seconds" % (end-start))
        print("mean loss: %.4f" % (total_loss / (train_input.shape[0] // BATCH_SIZE)))
        torch.save(model.state_dict(), "cbow.epoch{}.model".format(epoch))

    # Here you need to save the model's hidden layer which is V * D word embedding matrix.
    # Then, use the word embedding matrix to get vectors for word

if __name__ == '__main__':
    main()