import numpy, random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

from TrainModel import TrainModel 

#This function will do the bag of word process to the userInput
#It will check what words in userInput is in allWords which we got from the Json file 
def bagOfWords(userInput, allWords):
    #The bag in this function will the the length of the allWords list 
    bag = [0 for i in range(len(allWords))]
    userInputWords = [stemmer.stem(word.lower()) for word in nltk.word_tokenize(userInput)]
    
    #For all the stemmed words in userInput, this loop checks to see if the word is in allWords, 
    for word in userInputWords:
        for i, w in enumerate(allWords):
            if w == word: 
                bag[i] = 1 
    #Returneds a converted bag list to a numpy array which is easier for the model to process
    return numpy.array(bag)

def main():
    #Gets all the needed data from the training model functoin 
    model, allWords, tags, data = TrainModel()
    print("Welcome to the tech shop! My name is Eliza! Starting asking questions to talk to me (type quit to stop)!")
    name = input("What is your name? ")
    print(f'Eliza: Hello {name}, Nice to meet you! How can I help you today?')
  
    while True:
        userInput = input(f'{name}: ')
        if userInput.lower() == 'quit':
            break 
        #Model predicts the probability of the array being a tag for each tag
        results = model.predict([bagOfWords(userInput, allWords)])
        #Gets the index of highest probablity of a certain tag
        resultsIndex = numpy.argmax(results)
        #Uses the index to get the actual tag 
        tag = tags[resultsIndex]
       
       #For the chosen tag, a random response is chosen 
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print('Eliza: ', random.choice(responses))

main()