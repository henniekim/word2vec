import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import time
import collections

from fileIO import *
torch.manual_seed(1)

def main():
    EMBEDDING_DIM = 100
    if torch.cuda.is_available():
        VOCAB_SIZE = 30000
    else:
        VOCAB_SIZE = 30000

    file = Files()
    file.openQuestion()



    UNK_TOKEN = "<UNK>"
    WINDOW_SIZE = 5
    BATCH_SIZE = 2048

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
    pass
    ##
    if torch.cuda.is_available():
        model = torch.load('cbow.epoch5.model')
    else :
        model = torch.load('cbow.epoch5.model',  map_location='cpu')

    data = model['embeddings.weight']

    # 편하게 target vector를 얻기 위해 함수 정의
    def word2vec(word):
        return data[word_to_ix[word]]

    wordList = []

    for key, value in word_to_ix.items():
        wordList.append([key, value])

    def arg2word(argmax):
        return wordList[argmax]

    count = 0
    correct = 0

    for inferenceIdx in range(len(file.questionWord)):
        if len(file.questionWord[inferenceIdx]) < 3 : # text 파일 중에 단어가 4개가 아닌 줄이 있음.
            continue
        target = file.questionWord[inferenceIdx][0]
        minus = file.questionWord[inferenceIdx][2]
        plus = file.questionWord[inferenceIdx][3]
        gt = file.questionWord[inferenceIdx][1]

        try :
            targetVec = word2vec(target.lower()) # test 파일 중에 대문자를 소문자로 바꾸어줌
            minusVec = word2vec(minus.lower())
            plusVec = word2vec(plus.lower())
            gtVec = word2vec(gt.lower())

        except :
            print("unknown!")
            continue

        result = targetVec - minusVec + plusVec

        if torch.cuda.is_available() :
            zeros = torch.tensor((), dtype=torch.float32).cuda()
        else :
            zeros = torch.tensor((), dtype=torch.float32)

        zeros = zeros.new_zeros((VOCAB_SIZE + 1, EMBEDDING_DIM))
        broadcasted_target = torch.add(zeros, result)

        norm_target = torch.nn.functional.normalize(broadcasted_target)
        norm_data = torch.nn.functional.normalize(data)

        cosine_similarity_result = torch.nn.functional.cosine_similarity(norm_data, broadcasted_target)
        # topk_result = torch.topk(cosine_similarity_result, k=5)
        #
        # index0 = topk_result.indices[0]
        # index1 = topk_result.indices[1]
        # index2 = topk_result.indices[2]
        # index3 = topk_result.indices[3]
        # index4 = topk_result.indices[4]
        #
        # argword0 = arg2word(index0)
        # argword1 = arg2word(index1)
        # argword2 = arg2word(index2)
        # argword3 = arg2word(index3)
        # argword4 = arg2word(index4)



        # print(topk_result)
        argmax2 = torch.argmax(cosine_similarity_result)

        # pairwise_distance = torch.nn.PairwiseDistance(keepdim=False)
        # pairwise_distance_result = pairwise_distance(broadcasted_target, data)
        # argmax = torch.argmax(pairwise_distance_result)

        argword2 = arg2word(argmax2)
        gt = gt.lower()

        # argword = arg2word(argmax)

        if argword2[0]  == gt :
            print('correct ! ')
            print('guess : '+argword2[0]+' answer : '+gt)
            count += 1
            correct += 1

        else :
            print('wrong ! ')
            print('guess : '+argword2[0]+' answer : '+gt)
            count += 1

    print("맞춘 개수 : "+str(correct)+" / "+str(count))
    print(str(correct/count*100)+'%')

if __name__ == '__main__':
  main()