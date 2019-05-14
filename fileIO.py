import os

class Files :
    def __init__(self):
        pass

    def openQuestion(self):
        file= open("questions_words.txt", "r")
        self.question = file.read()
        self.question = self.question.split('\n')
        self.questionWord = []
        for questionIdx in range (1, len(self.question)):
            self.questionWord.append(self.question[questionIdx].split(' '))
        print("Question Word 를 불러왔습니다.")

        pass