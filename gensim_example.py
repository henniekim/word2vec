import gensim
import torch

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('model loaded!')



EMBEDDING_DIM = 300
VOCAB_SIZE = 3000000

file = open("questions_words.txt", "r")
question = file.read()
question = question.split('\n')
questionWord = []
for questionIdx in range(1, len(question)):
    questionWord.append(question[questionIdx].split(' '))
print("Question Word 를 불러왔습니다.")

def word2vec(word):
    return model.vectors[model.vocab[word].index]

def arg2word(arg):
    return model.index2word[arg]

count = 0
correct = 0

data = torch.from_numpy(model.vectors)

for inferenceIdx in range(len(questionWord)):
    if len(questionWord[inferenceIdx]) < 3:  # text 파일 중에 단어가 4개가 아닌 줄이 있음.
        continue


    target = questionWord[inferenceIdx][1]
    minus = questionWord[inferenceIdx][0]
    plus = questionWord[inferenceIdx][2]
    gt = questionWord[inferenceIdx][3]

    result = model.most_similar(positive=[target, plus], negative=[minus])
    print(result)

    # try:
    #     targetVec = word2vec(target.lower())  # test 파일 중에 대문자를 소문자로 바꾸어줌
    #     minusVec = word2vec(minus.lower())
    #     plusVec = word2vec(plus.lower())
    #     gtVec = word2vec(gt.lower())
    #
    # except:
    #     print("unknown!")
    #     continue
    #
    # result = targetVec - minusVec + plusVec
    # result = torch.from_numpy(result)
    #
    # if torch.cuda.is_available():
    #     zeros = torch.tensor((), dtype=torch.float32)
    # else:
    #     zeros = torch.tensor((), dtype=torch.float32)
    #
    # zeros = zeros.new_zeros((VOCAB_SIZE, EMBEDDING_DIM))
    # zeros = torch.add(zeros, result)
    #
    # cosine_similarity_result = torch.nn.functional.cosine_similarity(zeros, data)
    #
    # argmax2 = torch.argmax(cosine_similarity_result)
    # argword2 = arg2word(argmax2)
    gt = gt.lower()
    result_ = result[0][0].lower()
    #
    if result_ == gt:
        print('correct ! ')
        print('guess : ' + result_ + 'answer : ' + gt)
        count += 1
        correct += 1

    else:
        print('wrong ! ')
        print('guess : ' + result_ + ' answer : ' + gt)
        count += 1

print("맞춘 개수 : " + str(correct) + " / " + str(count))

pass