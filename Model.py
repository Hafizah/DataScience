#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries for NLP and TensorFlow

# In[1]:


#important libraries
import numpy as np # provides fast mathematical function processing
import tensorflow as tf # machine learning framework
from tensorflow.keras.models import Sequential # for plain layers where each layer has exactly one input tensor and one output tensor
from tensorflow.keras.layers import Dense, Dropout # regular densely-connected neural network layer, applies dropout to the input
from tensorflow.keras.preprocessing.text import Tokenizer # vectorize text into integers
import random # generate random numbers



# ### Load the Data

# In[2]:


#load chatbot intents 

import json
with open('Chatbot_Intents.json') as file:
  data=json.load(file)


# ### Text Pre-Processing with NLTK

# In[3]:


# Initiate stemming object
# NLP:for example -- "roaster", "roasting", "roasts" ---> "roast"

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()


# In[4]:




# consist of unique stemmed words/tokens from patterns extended in this list. No duplicates
words = []
# consist of tag words from intent
labels = []
# consist of tokenized sentences from patterns appended in this list
doc_x = []
# consists of tag words from intent matching tokens in doc_x
doc_y = []

# loop through each sentences in the data/intent
for intent in data['intents']:
    # loop through each sentences in patterns in intent
    for pattern in intent['patterns']:
        # tokenize each words in the pattern in intent
        wrds = nltk.word_tokenize(pattern)
        # method iterates over its argument adding each element to the list by extending the list
        words.extend(wrds)
        # method adds its argument as a single element to the end of a list. Length of the list increase by one
        doc_x.append(wrds)
        doc_y.append(intent['tag'])
        
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# stems and lower case the words 
words = [stemmer.stem(w.lower()) for w in words if w != '?']
 
# set() removes duplicates, list() change into a list and sorted() sort in ascending order
words = sorted(list(set(words)))

labels = sorted(labels)


# ### Transformation of Text in the Corpus to Vector of Numbers as Input to ML Model

# In[5]:


# creating training data from corpus. Change texts into array of numbers
# Bag of words (Bow) is a method to extract features from text documents. These features can be used to train ML model. 
# Bow creates a vocabulary of all the unique words in documents in the training set
# Bow disregards order in which they appear

X_train = []
y_train = []

# empty array for output
out_empty = [0 for _ in range(len(labels))]

# create bag of words for each sentences 
for x, doc in enumerate(doc_x):
    # initialize bag of words
    bag = []
    # stem and change all words to lower case
    wrds = [stemmer.stem(w.lower()) for w in doc]
    # use for loop to create an array of bag of words
    for w in words:
        bag.append(1) if w in wrds else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = out_empty[:]
    output_row[labels.index(doc_y[x])] = 1

    # result of 'bag' added to training list
    X_train.append(bag)
    # result of 'output_row' added to output list
    y_train.append(output_row)

# change to numpy array
X_train = np.array(X_train)
y_train = np.array(y_train)


# ### Create Neural Network

# In[6]:


# build model architecture
# dense 128 ---> unit or number of neurons
# droupout layers with rate of 0.5 are added to "turn off" neurons during training to prevent overfitting
# The length of teh vector = vocabulary size (how many unique words in the document without duplicates)
# categorical crossentropy loss function is used in multi-class classification tasks 

model=Sequential()
model.add(Dense(128, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[7]:


# summarize the architecture of the model
model.summary()


# In[8]:


# train model
model.fit(X_train, y_train, epochs=700, batch_size=5)





# In[15]:


model.save("final_model.h5")





# In[ ]:




