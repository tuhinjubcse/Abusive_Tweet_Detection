import numpy as np
import pandas as pd
from collections import defaultdict
import re
import sys
import os
from keras.constraints import maxnorm
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
from keras.layers import Conv1D, MaxoutDense,MaxPooling1D,GaussianNoise,InputLayer, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed,Activation
from keras.models import Sequential ,Model
import sys
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializations
from sklearn.metrics import classification_report , precision_recall_fscore_support,precision_score,recall_score,f1_score,confusion_matrix
import random
import pdb
from string import punctuation
import math
from my_tokenizer import glove_tokenize
from collections import defaultdict
from data_handler import get_data
from keras.regularizers import l2
from keras import regularizers
from keras import constraints
from sklearn.feature_extraction.text import TfidfVectorizer
import glob


word2vec_model = None
freq = defaultdict(int)
vocab, reverse_vocab = {}, {}
EMBEDDING_DIM = 300
tweets = {}
MAX_SENT_LENGTH = 0
MAX_SENTS = 0
batch_size = 256
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.1
INITIALIZE_WEIGHTS_WITH = 'glove'
SCALE_LOSS_FUN = False
n_gram_features_num = 1
liwc_features_num = 18


class AttLayer(Layer):
    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.init = initializations.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None 

    def call(self, x, mask=None):
        uit = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = K.squeeze(K.dot(uit, K.expand_dims(self.u)), axis=-1)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        #list_of_outputs = [K.sum(weighted_input, axis=1), a]
        return K.sum(weighted_input, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])

def batch_gen(X1, X2): # X1 and X2 have same number of samples. Batch created will be using the same indices.
    n_batches = X1.shape[0]/float(batch_size)
    n_batches = int(math.ceil(n_batches))
    end = int(X1.shape[0]/float(batch_size)) * batch_size

    n = 0
    for i in range(0,n_batches):
        if i < n_batches - 1: 
            batch1 = X1[i*batch_size:(i+1) * batch_size, :]
            batch2 = X2[i*batch_size:(i+1) * batch_size, :]
            yield batch1, batch2
        
        else:
            batch1 = X1[end: , :]
            batch2 = X2[end: , :]
            n += X1[end:, :].shape[0]

            yield batch1, batch2

def get_tfidf_features():
    tweets = get_data() # getting list of tweets (each tweet in a map format with keys text, label and user)
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    flag = True
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower()) # tokenizing like converting # into <hashtag> etc.
        text = ' '.join([c for c in text if c not in punctuation]) # removing punctuation
        X.append(text)
        y.append(y_map[tweet['label']])
    tfidf_transformer = TfidfVectorizer(ngram_range=(1,2), analyzer='word',stop_words='english',max_features=5000)
    X_tfidf = tfidf_transformer.fit_transform(X)
    print(X_tfidf.shape)

    return X_tfidf, np.array(y)

def get_embedding(word):
    try:
        return word2vec_model[word]
    except (ValueError, KeyError) as e:
        print('Encoding not found: %s' %(word))
        return np.zeros(EMBEDDING_DIM)

def get_embedding_weights():
    embedding = np.zeros((len(vocab) + 1, EMBEDDING_DIM))
    n = 0
    print(len(vocab))
    for k, v in vocab.items():
        try:
            embedding[v] = word2vec_model[k]
        except (ValueError, KeyError) as e:
            n += 1
            pass
    print("%d embedding missed"%n)
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
        seq, _emb = [], []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))  # what does vocab.get(word, length) do here, isn't vocab a dict?
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y

# 1-d feature per tweet which counts the number of abusive words in a tweet
def getAbusiveFeatures():
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }
    f = open('abusive_dict.txt','r')
    m = {}
    for line in f:
        line = line.strip()
        m[line]=True
    tweets = get_data()
    X, y = [], []
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower()) # does it correct spelling as well?
        c = 0
        for word in text:
            if word in m:
                c = c+1
        X.append([c])
        y.append(y_map[tweet['label']])
    return np.array(X),np.array(y)

def get_liwc_features_from_text():
    filenames = glob.glob("./LIWC_features/*.csv")
    print(filenames)
    y_map = {
            'none': 0,
            'racism': 1,
            'sexism': 2
            }
    tweets = get_data()
    X, y = [], []
    # create a dict of lists of words in all liwc files
    features_dict = {}
    for file in filenames:
        f = open(file,'r')
        m = {}
        for line in f:
            line = line.strip()
            m[line]=True
        features_dict[file] = m
    
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower())
        features = []
        for file in filenames:
            c = 1
            for word in text:
                if any([word.startswith(s) for s in features_dict[file]]):
                    c = c+1
            features.append(c)
        X.append(features)
        y.append(y_map[tweet['label']])

    # normalised results
    X = np.array(X)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    return X, np.array(y)

def select_tweets():
    # selects the tweets as in mean_glove_embedding method
    # Processing
    tweets = get_data()
    tweet_return = []
    c = 1
    for tweet in tweets:
        _emb = 0
        words = glove_tokenize(tweet['text'].lower())
        for w in words:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        c = c+1
        # if _emb:   # Not a blank tweet
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

        for word in words:
            if word not in vocab:
                vocab[word] = vocab_index
                reverse_vocab[vocab_index] = word
                vocab_index += 1
            freq[word] += 1
    vocab['UNK'] = len(vocab) + 1
    reverse_vocab[len(vocab)] = 'UNK'  

def shuffle_weights(model):
    weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)


def lstm_model(sentence_len,embeddings_matrix, embedding_dim, X_handcrafted):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model1 = Sequential()
    model1.add(Embedding(len(vocab)+1, embedding_dim, weights=[embeddings_matrix] ,input_length=sentence_len, trainable=True))
    model1.add(Dropout(0.25))
    model1.add(Bidirectional(LSTM(150,return_sequences=True)))
    model1.add(Dropout(0.25))
    model1.add(Bidirectional(LSTM(150,return_sequences=True)))
    model1.add(Dropout(0.25))
    model1.add(AttLayer())
    print("model1")
    print(model1.summary())

    model2 = Sequential()
    model2.add(InputLayer(input_shape=(X_handcrafted.shape[1],)))
    print("model2")
    print(model2.summary())

    model = Sequential()
    model.add(Merge([model1,model2], mode='concat'))
    # model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
    # model.add(Dropout(0.25))
    model.add(Dense(3,activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print("merged model")
    print(model.summary())

    return model


def train_LSTM(X, y, X_handcrafted, y_handcrafted,inp_dim, weights, epochs=12, batch_size=512):

    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    print(cv_object)
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    print("check......... - ", sentence_len)
    c = 1
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model = lstm_model(sentence_len, weights, EMBEDDING_DIM, X_handcrafted)
        else:
            print("ERROR!")
            return
        X_train1, y_train1 = X[train_index], y[train_index]
        X_test1, y_test1 = X[test_index], y[test_index]
        y_train1 = y_train1.reshape((len(y_train1), 1))
        X_temp1 = np.hstack((X_train1, y_train1))

        X_train2, y_train2 = X_handcrafted[train_index], y_handcrafted[train_index]
        X_test2, y_test2 = X_handcrafted[test_index], y_handcrafted[test_index]
        y_train2 = y_train2.reshape((len(y_train2), 1))
        X_temp2 = np.hstack((X_train2, y_train2))

        f = open('./CV/cv_'+str(c)+'_liwc_features.txt','w+')
        for q in range(len(X_train2)):
            f.write(str(X_handcrafted[q]) + '\n')


        for epoch in range(epochs):
            print('Epoch ',epoch,'\n')
            for X_batch1, X_batch2 in batch_gen(X_temp1, X_temp2):
                curr_X1 = X_batch1[:, :sentence_len]
                curr_Y1 = X_batch1[:, sentence_len]
                curr_X2 = X_batch2[:, :liwc_features_num]
                curr_Y2 = X_batch2[:, liwc_features_num]
                class_weights = None

                try:
                    curr_Y1 = to_categorical(curr_Y1, nb_classes=3)
                    curr_Y2 = to_categorical(curr_Y2, nb_classes=3)
                except ValueError as e:
                    print(e)
                                                    
                loss, acc = model.train_on_batch([curr_X1,curr_X2], curr_Y1, class_weight=None)
        y_pred = model.predict_on_batch([X_test1, X_test2])
        y_pred = np.argmax(y_pred, axis=1)

        f = open('./CV/cv_'+str(c)+'_labels.txt','w+')
        for q in range(len(y_pred)):
            f.write(str(y_pred[q])+'\t'+str(y_test1[q])+'\n')

        c = c+1
        print(classification_report(y_test1, y_pred))
        print(precision_recall_fscore_support(y_test1, y_pred))
        print(confusion_matrix(y_test1, y_pred))
        # print(y_pred)
        p += precision_score(y_test1, y_pred, average='weighted')
        r += recall_score(y_test1, y_pred, average='weighted')
        f1 += f1_score(y_test1, y_pred, average='weighted')

    NO_OF_FOLDS = 10
    print("weighted results are")
    print("average precision is %f" %(p/NO_OF_FOLDS))
    print("average recall is %f" %(r/NO_OF_FOLDS))
    print("average f1 is %f" %(f1/NO_OF_FOLDS))


np.random.seed(42)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./datastories.twitter.300d.txt')
tweets = select_tweets()
gen_vocab()
X, y = gen_sequence()
# X_tfidf , y_tfidf = get_tfidf_features()
#X_abs , y_abs = getAbusiveFeatures()
# X_tfidf = X_tfidf.todense()
X_liwc, y_liwc = get_liwc_features_from_text()

MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
print("max seq length is %d"%(MAX_SEQUENCE_LENGTH))

data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(y)
data, y, X_handcrafted, y_handcrafted = sklearn.utils.shuffle(data, y,X_liwc, y_liwc)
W = get_embedding_weights()
train_LSTM(data, y, X_liwc, y_liwc, EMBEDDING_DIM, W)