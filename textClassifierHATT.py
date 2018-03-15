# author - Richard Liao 
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from string import maketrans

import sys
import os

os.environ['KERAS_BACKEND']='theano'

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
import sys
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations

reload(sys)
sys.setdefaultencoding('utf8')

MAX_SENT_LENGTH = 0
MAX_SENTS = 0
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1

def is_ascii(text):
        try:
            text.encode('ascii')
            return True
        except:
            return False     

def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]
    
keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

from nltk import tokenize

tweets = []
labels = []
texts = []
m = {'racism':0,'sexism':1,'none':2}

f = open('tokenized_tweets.txt','r')

for line in f:
    line = line.strip().split()
    label = line[-1]
    line.pop()
    text = ' '.join(line).decode("utf-8")
    texts.append(text)
    sentences = tokenize.sent_tokenize(text)
    MAX_SENTS = max(MAX_SENTS,len(sentences))
    for sent in sentences:
        MAX_SENT_LENGTH = max(MAX_SENT_LENGTH,len(sent.split()))
    tweets.append(sentences)
    labels.append(m[label])

tokenizer = Tokenizer(filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(texts)


data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

for i, sentences in enumerate(tweets):
    for j, sent in enumerate(sentences):
        if j< MAX_SENTS:
            wordTokens = text_to_word_sequence(sent)
            k=0
            for _, word in enumerate(wordTokens):
                if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
                    data[i,j,k] = tokenizer.word_index[word]
                    k=k+1                    

word_index =  tokenizer.word_index                   
print('Total %s unique tokens.' % len(word_index))

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]

# print('Number of positive and negative reviews in traing and validation set')
# print y_train.sum(axis=0)
# print y_val.sum(axis=0)
# racism = 0
# sexism = 0
# none = 0

# for y in y_train:
#     if y[2]==0 and y[1]==0:
#         racism = racism +1
#     elif y[1]==0 and y[2]==1:
#         sexism = sexism+1
#     elif y[1]==1 and y[2]==0:
#         none = none+1

# print racism , sexism, none



count = 0
GLOVE_DIR = "./"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'datastories.twitter.300d.txt'))
for line in f:
    values = line.split()
    coefs = np.asarray(values[1:], dtype='float32')
    word = values[0]
    if is_ascii(word)==False and (word.decode('utf8') in tokenizer.word_index):
        count = count+1
        embeddings_index[word.decode('utf8')] = coefs
    elif word in tokenizer.word_index:
        count = count+1
        embeddings_index[word] = coefs
    else:
        embeddings_index[word] = coefs
    
    
f.close()

print 'Known words',count

print('Total %s word vectors.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SENT_LENGTH,
                            trainable=True)

sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(LSTM(100))(embedded_sequences)
sentEncoder = Model(sentence_input, l_lstm)

tweet_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
tweet_encoder = TimeDistributed(sentEncoder)(tweet_input)
l_lstm_sent = Bidirectional(LSTM(100))(tweet_encoder)
preds = Dense(3, activation='softmax')(l_lstm_sent)
model = Model(tweet_input, preds)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

print("model fitting - Hierachical LSTM")
print model.summary()
model.fit(x_train, y_train, validation_data=(x_test, y_test),
          nb_epoch=10, batch_size=50)

# # building Hierachical Attention network
# embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
# for word, i in word_index.items():
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector
        
# embedding_layer = Embedding(len(word_index) + 1,
#                             EMBEDDING_DIM,
#                             weights=[embedding_matrix],
#                             input_length=MAX_SENT_LENGTH,
#                             trainable=True)

# class AttLayer(Layer):
#     def __init__(self, **kwargs):
#         self.init = initializations.get('normal')
#         #self.input_spec = [InputSpec(ndim=3)]
#         super(AttLayer, self).__init__(**kwargs)

#     def build(self, input_shape):
#         assert len(input_shape)==3
#         #self.W = self.init((input_shape[-1],1))
#         self.W = self.init((input_shape[-1],))
#         #self.input_spec = [InputSpec(shape=input_shape)]
#         self.trainable_weights = [self.W]
#         super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

#     def call(self, x, mask=None):
#         eij = K.tanh(K.dot(x, self.W))
        
#         ai = K.exp(eij)
#         weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
#         weighted_input = x*weights.dimshuffle(0,1,'x')
#         return weighted_input.sum(axis=1)

#     def get_output_shape_for(self, input_shape):
#         return (input_shape[0], input_shape[-1])

# sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sentence_input)
# l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
# l_dense = TimeDistributed(Dense(200))(l_lstm)
# l_att = AttLayer()(l_dense)
# sentEncoder = Model(sentence_input, l_att)

# review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
# review_encoder = TimeDistributed(sentEncoder)(review_input)
# l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
# l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
# l_att_sent = AttLayer()(l_dense_sent)
# preds = Dense(2, activation='softmax')(l_att_sent)
# model = Model(review_input, preds)

# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['acc'])

# print("model fitting - Hierachical attention network")
# model.fit(x_train, y_train, validation_data=(x_val, y_val),
#           nb_epoch=10, batch_size=50)