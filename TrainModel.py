import json
import random
import tensorflow
import tflearn
import numpy
import nltk
from nltk.stem.lancaster import LancasterStemmer

stemmer = LancasterStemmer()
nltk.download('punkt')

def TrainModel():

    with open('intents.json') as file:
        data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wordList = nltk.word_tokenize(pattern)
            words.extend(wordList)
            docs_x.append(wordList)
            docs_y.append(intent['tag'])

        if intent['tag'] not in labels:
            labels.append(intent['tag'])
    print(
        f'This is words: {words}\nThis is labels: {labels}\nThis is docs_x: {docs_x}\nThis is docs_y: {docs_y}')
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    training = []
    output = []

    outEmpty = [0 for i in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wordList = [stemmer.stem(w.lower()) for w in doc]
        print(f'This is wordList: {wordList}')
        for w in words:
            if w in wordList:
                bag.append(1)
            else:
                bag.append(0)
        print(f'This is outEmpty: {outEmpty[:]}')
        outputRow = outEmpty[:]
        outputRow[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(outputRow)
        print(f'This is outputRow: {outputRow}')
        print(f'This is bag: {bag}')

    print(f'This is training: {training}\nThis is output: {output}')

    tensorflow.compat.v1.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(output[0]), activation='softmax')
    net = tflearn.regression(net)

    # This now trains the model and saves the model to a file
    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')

    return model, words, labels, data 
