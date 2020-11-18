import numpy, random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from TrainModel import TrainModel 

def bagOfWords(userInput, allWords):
    bag = [0 for i in range(len(allWords))]
    userInputWords = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(userInput)]
    print(f'This is sWords: {userInputWords}')
    for word in userInputWords:
        for i, w in enumerate(allWords):
            if w == word: 
                bag[i] = 1 
    print(f'This is the numpy array: {numpy.array(bag)}')
    return numpy.array(bag)

def main():
    model, allWords, labels, data = TrainModel()
    print("Welcome to the tech shop! My name is Eliza! Starting asking questions to talk to me (type quit to stop)!")
    name = input("What is your name? ")
    print(f'Eliza: Hello {name}, Nice to meet you! How can I help you today?')
  
    while True:
        userInput = input(f'{name}: ')
        if userInput.lower() == 'quit':
            break 

        results = model.predict([bagOfWords(userInput, allWords)])
        resultsIndex = numpy.argmax(results)
        tag = labels[resultsIndex]
       
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print('Eliza: ', random.choice(responses))

main()