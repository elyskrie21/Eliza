#This the code for the machine learning model that is built form tensorflow and the higher level api tftlearn because I am not very good a machine learning yet

#For this to work properly and to see the model being trained, you need to use pip to install tensorflow, tflearn, and nltk
import json
import random
import tensorflow
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer

#I am using the Lancaster method of stemming words, which is the process of finding the root of the word 
stemmer = LancasterStemmer()
nltk.download('punkt')

#Function to train the model which will be called in the Main python file 
def TrainModel():

    #Using a json file because of the easy formatting of tags, patterns and responses 
    with open('intents.json') as file:
        data = json.load(file)

    #Initializing, allWords will be a list of all the indivual unique words 
    #tags will be a list of all the tags in the json file
    #allPatterns will be a list containing lists of all the patterns tokenized 
    #tagsForEachPattern will corrospond a tag for all the patterns in allPatterns   
    allWords = []
    tags = []
    allPatterns = []
    tagsForEachPattern = []

    #This for loop will populate the 4 previous lists
    for intent in data['intents']:
        
        #Loops through each pattern in the Json file invidualy 
        for pattern in intent['patterns']:
            #Each pattern is tokenized, or in other words, divides a string into a list of word substrings 
            wordList = nltk.word_tokenize(pattern)
            #Using the extend method, wordList is iterated over to add each element into allWords 
            allWords.extend(wordList)
            #The entire tokenized pattern list is added to allPatterns as a single element 
            allPatterns.append(wordList)
            #Gets the tag for the pattern and adds it 
            tagsForEachPattern.append(intent['tag'])

        #this creates a list of tags where each tag is unique 
        if intent['tag'] not in tags:
            tags.append(intent['tag'])

    #The allWords list is turned lowercase and stemmed removing the questions marks. Then its sorted into a list of unique words 
    allWords = [stemmer.stem(w.lower()) for w in allWords if w != '?']
    allWords = sorted(list(set(allWords)))

    #These two list are going to be used to train our model 
    #The training list is all the data the model is going to use for its predictions
    #The output list will get the tag that the model should predict for each pattern 
    training = []
    output = []

    #A list of 0 the spans the length of the tags list which will be used during our Bag of Words process 
    outEmpty = [0 for i in range(len(tags))]

    #This for loop is runnnig the Bag of Words process because a model needs to trained using numbers and not strings
    for index, pattern in enumerate(allPatterns):
        #This bag list will become the length of allWords and will be made up of 0s and 1s
        #If a word in the pattern is in allWords then a 1 is appended, else a 0 is appened 
        bag = []

        #This created a list of stemmed words from the pattern to compare to the stemmed words in allWords 
        wordList = [stemmer.stem(w.lower()) for w in pattern]
        for word in allWords:
            if word in wordList:
                bag.append(1)
            else:
                bag.append(0)
        
        #outputRow is the length of tags with all 0s except for one 1. This 1 is what tag relates to the pattern of the bag list
        outputRow = outEmpty[:]
        outputRow[tags.index(tagsForEachPattern[index])] = 1

        #all the data is added to the training and output lists 
        training.append(bag)
        output.append(outputRow)
  
    #Setting up tensorflow and the neural networks
    tensorflow.compat.v1.reset_default_graph()

    #Brings the data into the system for processing 
    net = tflearn.input_data(shape=[None, len(training[0])])
    #Creates 3 more layers each with 8 "neurons"
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    #This creates the output layer after data is processed by the previous layers 
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    #This uses regressions to predict the tag "or the numerical value of the tag" by using previous data 
    net = tflearn.regression(net)

    # This now trains the model using DNN "Deep Neural Network" and saves the model to a file
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')

    return model, allWords, tags, data 