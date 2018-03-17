# author - Richard Liao 
# Dec 26 2016
import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from string import maketrans
from sklearn.model_selection import KFold
import sys
import os
import sklearn
from gensim.parsing.preprocessing import STOPWORDS
os.environ['KERAS_BACKEND']='theano'
import gensim
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras
from sklearn.model_selection import KFold
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Activation
from keras.models import Sequential ,Model
import sys
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from sklearn.metrics import classification_report , precision_recall_fscore_support,precision_score,recall_score,f1_score
import random
import pdb
from string import punctuation
import math
from my_tokenizer import glove_tokenize
from collections import defaultdict
from data_handler import get_data

reload(sys)
sys.setdefaultencoding('utf8')

word2vec_model = None
freq = defaultdict(int)
vocab, reverse_vocab = {}, {}
EMBEDDING_DIM = 25
tweets = {}
MAX_SENT_LENGTH = 0
MAX_SENTS = 0
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.1
INITIALIZE_WEIGHTS_WITH = 'random'
SCALE_LOSS_FUN = True


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializations.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))
        
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')
        
        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def batch_gen(X, batch_size):
    n_batches = X.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X.shape[0]/float(batch_size)) * batch_size
    n = 0
    for i in xrange(0,n_batches):
        if i < n_batches - 1: 
            batch = X[i*batch_size:(i+1) * batch_size, :]
            yield batch
        
        else:
            batch = X[end: , :]
            n += X[end:, :].shape[0]
            yield batch


def get_embedding(word):
    #return
    try:
        return word2vec_model[word]
    except Exception, e:
        print 'Encoding not found: %s' %(word)
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    print len(vocab)
    for k, v in vocab.iteritems():
        try:
            embedding[v] = word2vec_model[k]
        except Exception, e:
            n += 1
            pass
    print "%d embedding missed"%n
    return embedding


def gen_sequence():
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    flag = True
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y

def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    tweet_return = []
    for tweet in tweets:
        _emb = 0
        words = glove_tokenize(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            tweet_return.append(tweet)
    print('Tweets selected:', len(tweet_return))
    #pdb.set_trace()
    return tweet_return


def gen_vocab():
    # Processing
    vocab_index = 1
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower())
        text = ' '.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'

def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

def lstm_model(sequence_length, embedding_dim):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model = Sequential()
    model.add(Embedding(len(vocab)+1, embedding_dim, input_length=sequence_length, trainable=True))
    # model.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model.add(LSTM(50))
    model.add(Dropout(0.25))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()
    return model



def train_LSTM(X, y, model, inp_dim, weights, epochs=10, batch_size=512):
    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    print cv_object
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model.layers[0].set_weights([weights])
        elif INITIALIZE_WEIGHTS_WITH == "random":
            shuffle_weights(model)
        else:
            print "ERROR!"
            return
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        y_train = y_train.reshape((len(y_train), 1))
        X_temp = np.hstack((X_train, y_train))
        for epoch in xrange(epochs):
            print('Epoch ',epoch,'\n')
            for X_batch in batch_gen(X_temp, 512):
                x = X_batch[:, :sentence_len]
                y_temp = X_batch[:, sentence_len]

                class_weights = None
                if SCALE_LOSS_FUN:
                    class_weights = {}
                    class_weights[0] = np.where(y_temp == 0)[0].shape[0]/float(len(y_temp))
                    class_weights[1] = np.where(y_temp == 1)[0].shape[0]/float(len(y_temp))
                    class_weights[2] = np.where(y_temp == 2)[0].shape[0]/float(len(y_temp))

                try:
                    y_temp = to_categorical(y_temp, nb_classes=3)
                except Exception as e:
                    print e
                    print y_temp
                loss, acc = model.train_on_batch(x, y_temp, class_weight=class_weights)
                # print loss, acc

        y_pred = model.predict_on_batch(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        print classification_report(y_test, y_pred)
        print precision_recall_fscore_support(y_test, y_pred)
        # print y_pred
        p += precision_score(y_test, y_pred, average='weighted')
        p1 += precision_score(y_test, y_pred, average='micro')
        r += recall_score(y_test, y_pred, average='weighted')
        r1 += recall_score(y_test, y_pred, average='micro')
        f1 += f1_score(y_test, y_pred, average='weighted')
        f11 += f1_score(y_test, y_pred, average='micro')

    NO_OF_FOLDS = 10
    print "macro results are"
    print "average precision is %f" %(p/NO_OF_FOLDS)
    print "average recall is %f" %(r/NO_OF_FOLDS)
    print "average f1 is %f" %(f1/NO_OF_FOLDS)

    print "micro results are"
    print "average precision is %f" %(p1/NO_OF_FOLDS)
    print "average recall is %f" %(r1/NO_OF_FOLDS)
    print "average f1 is %f" %(f11/NO_OF_FOLDS)


np.random.seed(42)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./glove.twitter.27B.25d.txt')
tweets = select_tweets()
gen_vocab()
X, y = gen_sequence()
MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
print "max seq length is %d"%(MAX_SEQUENCE_LENGTH)

data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(y)
data, y = sklearn.utils.shuffle(data, y)
W = get_embedding_weights()
model = lstm_model(data.shape[1], EMBEDDING_DIM)
train_LSTM(data, y, model, EMBEDDING_DIM, W)





# def is_ascii(text):
#         try:
#             text.encode('ascii')
#             return True
#         except:
#             return False     

# def text_to_word_sequence(text,
#                           filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n',
#                           lower=True, split=" "):
#     if lower: text = text.lower()
#     if type(text) == unicode:
#         translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
#     else:
#         translate_table = maketrans(filters, split * len(filters))
#     text = text.translate(translate_table)
#     seq = text.split(split)
#     return [i for i in seq if i]
    
# keras.preprocessing.text.text_to_word_sequence = text_to_word_sequence

# from nltk import tokenize

# tweets = []
# labels = []
# texts = []
# m = {'racism':0,'sexism':1,'none':2}

# f = open('tokenized_tweets.txt','r')
# flag = False
# for line in f:
#     line = line.strip().split()
#     label = line[-1]
#     line.pop()
#     text = ' '.join(line).decode("utf-8")
#     texts.append(text)
#     sentences = tokenize.sent_tokenize(text)
#     MAX_SENTS = max(MAX_SENTS,len(sentences))
#     for sent in sentences:
#         MAX_SENT_LENGTH = max(MAX_SENT_LENGTH,len(sent.split()))
#     tweets.append(sentences)
#     labels.append(m[label])

# tokenizer = Tokenizer(filters='!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n')
# tokenizer.fit_on_texts(texts)


# data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

# for i, sentences in enumerate(tweets):
#     for j, sent in enumerate(sentences):
#         if j< MAX_SENTS:
#             wordTokens = text_to_word_sequence(sent)
#             k=0
#             for _, word in enumerate(wordTokens):
#                 if k<MAX_SENT_LENGTH and tokenizer.word_index[word]<MAX_NB_WORDS:
#                     data[i,j,k] = tokenizer.word_index[word]
#                     k=k+1                    

# word_index =  tokenizer.word_index                   
# print('Total %s unique tokens.' % len(word_index))


# count = 0
# GLOVE_DIR = "./"
# embeddings_index = {}
# f = open(os.path.join(GLOVE_DIR, 'datastories.twitter.300d.txt'))
# for line in f:
#     values = line.split()
#     coefs = np.asarray(values[1:], dtype='float32')
#     word = values[0]
#     if is_ascii(word)==False and (word.decode('utf8') in tokenizer.word_index):
#         count = count+1
#         embeddings_index[word.decode('utf8')] = coefs
#     elif word in tokenizer.word_index:
#         count = count+1
#         embeddings_index[word] = coefs
#     else:
#         embeddings_index[word] = coefs
    
    
# f.close()

# print 'Known words',count

# print('Total %s word vectors.' % len(embeddings_index))

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

# sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
# embedded_sequences = embedding_layer(sentence_input)
# l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
# l_dense = TimeDistributed(Dense(200))(l_lstm)
# l_att = AttLayer()(l_dense)
# sentEncoder = Model(sentence_input, l_att)


# tweet_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
# tweet_encoder = TimeDistributed(sentEncoder)(tweet_input)
# l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(tweet_encoder)
# l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
# l_att_sent = AttLayer()(l_dense_sent)
# preds = Dense(3, activation='softmax')(l_att_sent)
# model = Model(tweet_input, preds)


# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['fmeasure'])


# NO_OF_FOLDS = 10
# SCALE_LOSS_FUN = True
# cv_object = KFold(n_splits=NO_OF_FOLDS, shuffle=True, random_state=42)
# p, r, f1 = 0., 0., 0.
# p1, r1, f11 = 0., 0., 0.
# # sentence_len = X.shape[1]
# epochs = 10
# a = cv_object.split(data)
# print model.summary()
# for train_index, test_index in cv_object.split(data):
#         X_train = []
#         y_train = []
#         X_test = []
#         y_test = []
#         for i in train_index:
#             X_train.append(data[i])
#             y_train.append(labels[i])
#         for i in test_index:
#             X_test.append(data[i])
#             y_test.append(labels[i])
#         X_train = np.array(X_train)
#         X_test = np.array(X_test)
#         y_train = np.array(y_train)
#         y_test = np.array(y_test)
#         try:
#             y_train = to_categorical(y_train)
#         except Exception as e:
#             print e
#         for epoch in xrange(epochs):
#                 class_weights = None
#                 if SCALE_LOSS_FUN:
#                     class_weights = {}
#                     class_weights[0] = np.where(y_train == 0)[0].shape[0]/float(len(y_train))
#                     class_weights[1] = np.where(y_train == 1)[0].shape[0]/float(len(y_train))
#                     class_weights[2] = np.where(y_train == 2)[0].shape[0]/float(len(y_train))
#                 # print class_weights
#                 # print len(X_train),len(y_train)
#                 model.fit(X_train, y_train,nb_epoch=1, batch_size=512,class_weight=class_weights)

#         print len(X_test),len(y_test)
#         y_pred = model.predict(X_test)
#         y_pred = np.argmax(y_pred, axis=1)
#         print classification_report(y_test, y_pred)
#         print precision_recall_fscore_support(y_test, y_pred)
#         pwe = precision_score(y_test, y_pred, average='weighted')
#         p += pwe
#         pmi = precision_score(y_test, y_pred, average='micro')
#         p1 += pmi
#         rwe = recall_score(y_test, y_pred, average='weighted')
#         r +=  rwe
#         rmi = recall_score(y_test, y_pred, average='micro')
#         r1 += rmi
#         fwe = f1_score(y_test, y_pred, average='weighted')
#         f1 += fwe
#         fmi = f1_score(y_test, y_pred, average='micro')
#         f11 += fmi
#         print 'precision ----> ', pwe , pmi
#         print 'recall ----> ', rwe , rmi
#         print 'f1 ----> ', fwe , fmi



# print "macro results are"
# print "average precision is %f" %(p/NO_OF_FOLDS)
# print "average recall is %f" %(r/NO_OF_FOLDS)
# print "average f1 is %f" %(f1/NO_OF_FOLDS)

# print "micro results are"
# print "average precision is %f" %(p1/NO_OF_FOLDS)
# print "average recall is %f" %(r1/NO_OF_FOLDS)
# print "average f1 is %f" %(f11/NO_OF_FOLDS)




