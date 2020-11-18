import numpy, random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from TrainModel import TrainModel 

def bagOfWords(s, words):
    bag = [0 for i in range(len(words))]
    sWords = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(s)]

    for se in sWords:
        for i, w in enumerate(words):
            if w == se: 
                bag[i] = 1 
    
    return numpy.array(bag)

def main():
    model, words, labels, data = TrainModel()
    print("Welcome to the tech shop! My name is Eliza! Starting asking questions to talk to me (type quit to stop)!")
    name = input("What is your name? ")
    print(f'Hello {name}, Nice to meet you! How can I help you today?')
  
    while True:
        userInput = input(f'{name}: ')
        if userInput.lower() == 'quit':
            break 

        results = model.predict([bagOfWords(userInput, words)])
        resultsIndex = numpy.argmax(results)
        tag = labels[resultsIndex]

        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
                print(tg['tag'])
                print(responses)
        
        print(random.choice(responses))

main()