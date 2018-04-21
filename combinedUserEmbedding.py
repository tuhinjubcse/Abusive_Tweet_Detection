import numpy as np
import pandas as pd
import cPickle
from collections import defaultdict
import re
from string import maketrans
from sklearn.model_selection import KFold
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

reload(sys)
sys.setdefaultencoding('utf8')

word2vec_model = None
freq = defaultdict(int)
vocab, reverse_vocab = {}, {}
EMBEDDING_DIM = 300
tweets = {}
MAX_SENT_LENGTH = 0
MAX_SENTS = 0
batch_size = 512
MAX_NB_WORDS = 20000
VALIDATION_SPLIT = 0.1
INITIALIZE_WEIGHTS_WITH = 'glove'
SCALE_LOSS_FUN = False
n_gram_features_num = 300
X_tweet = None


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
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def compute_mask(self, input, input_mask=None):
        return None 

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = K.dot(uit, self.u)
        a = K.exp(ait)
        if mask is not None:
            a *= K.cast(mask, K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
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

def getUserFeatures():
    X = []
    tweetsFrom = []
    tweet_to_user = {}
    for line in open('./tweet_user.txt'):
        user = line.strip().split('\t')[0]
        tweet = line.strip().split('\t')[1]
        tweet_to_user[tweet] = user
    for line in open('./shuffled_data.txt'):
        line = line.strip().split()
        line.pop()
        line = ' '.join(line)
        X.append(tweet_to_user[line])
        tweetsFrom.append(line)
    return np.asarray(X),tweetsFrom



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
        seq, _emb = [], []
        for word in text:
            seq.append(vocab.get(word, vocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y



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
        # words = [word for word in words if word not in STOPWORDS]

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




def lstm_model(sequence_length,embeddings_matrix, embedding_dim,X_tfidf):
    model_variation = 'LSTM'
    print('Model variation is %s' % model_variation)
    model1 = Sequential()
    model1.add(Embedding(len(vocab)+1, embedding_dim, weights=[embeddings_matrix] ,input_length=sequence_length, trainable=True))
    model1.add(Dropout(0.25))#, input_shape=(sequence_length, embedding_dim)))
    model1.add(Bidirectional(LSTM(150,return_sequences=True)))
    model1.add(Dropout(0.25))
    model1.add(Bidirectional(LSTM(150,return_sequences=True)))
    model1.add(Dropout(0.25))
    model1.add(AttLayer())
    print model1.summary()

    model2 = Sequential()
    model2.add(InputLayer(input_shape=(300,)))
    print model2.summary()

    model = Sequential()
    model.add(Merge([model1,model2], mode='concat'))
    model.add(MaxoutDense(100, W_constraint=maxnorm(2)))
    model.add(Dropout(0.25))
    model.add(Dense(3,activity_regularizer=l2(0.0001)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()

    return model





def train_LSTM(X, y, X_tfidf, y_tfidf,inp_dim, weights, epochs=10, batch_size=512):
    cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    print cv_object
    p, r, f1 = 0., 0., 0.
    p1, r1, f11 = 0., 0., 0.
    sentence_len = X.shape[1]
    c = 1
    for train_index, test_index in cv_object.split(X):
        if INITIALIZE_WEIGHTS_WITH == "glove":
            model = lstm_model(sentence_len,weights ,EMBEDDING_DIM,X_tfidf)
        else:
            print "ERROR!"
            return
        X_train1, y_train1 = X[train_index], y[train_index]
        X_test1, y_test1 = X[test_index], y[test_index]
        y_train1 = y_train1.reshape((len(y_train1), 1))
        X_temp1 = np.hstack((X_train1, y_train1))


        X_train2, y_train2 = X_tfidf[train_index], y_tfidf[train_index]
        X_test2, y_test2 = X_tfidf[test_index], y_tfidf[test_index]
        user_train_model = gensim.models.KeyedVectors.load_word2vec_format('./user_embeddings/user_train_'+str(c)+'.txt')
        user_test_model = gensim.models.KeyedVectors.load_word2vec_format('./user_embeddings/user_test_'+str(c)+'.txt')

        X_train_user = []
        X_test_user = []
        for k in range(len(X_train2)):
            X_train_user.append(user_train_model[X_train2[k]])
        for k in range(len(X_test2)):
            X_test_user.append(user_test_model[X_test2[k]])
        X_train_user = np.asarray(X_train_user)
        X_test_user = np.asarray(X_test_user)
        X_train2 = X_train_user
        X_test2 = X_test_user
        y_train2 = y_train2.reshape((len(y_train2), 1))
        X_temp2 = np.hstack((X_train2, y_train2))


        for epoch in xrange(epochs):
            print('Epoch ',epoch,'\n')
            for X_batch1, X_batch2 in batch_gen(X_temp1, X_temp2):
                curr_X1 = X_batch1[:, :sentence_len]
                curr_Y1 = X_batch1[:, sentence_len]
                curr_X2 = X_batch2[:, :n_gram_features_num]
                curr_Y2 = X_batch2[:, n_gram_features_num]
                class_weights = None

                try:
                    curr_Y1 = to_categorical(curr_Y1, nb_classes=3)
                    curr_Y2 = to_categorical(curr_Y2, nb_classes=3)
                except Exception as e:
                    print e
                loss, acc = model.train_on_batch([curr_X1,curr_X2], curr_Y1, class_weight=None)
        y_pred = model.predict_on_batch([X_test1, X_test2])
        y_pred = np.argmax(y_pred, axis=1)
        f = open('./CV/cv_'+str(c)+'_labels.txt','w')
        for q in range(len(y_pred)):
            f.write(str(y_pred[q])+'\t'+str(y_test1[q])+'\n')
        c = c+1
        print classification_report(y_test1, y_pred)
        print precision_recall_fscore_support(y_test1, y_pred)
        print confusion_matrix(y_test1, y_pred)
        # print y_pred
        p += precision_score(y_test1, y_pred, average='weighted')
        r += recall_score(y_test1, y_pred, average='weighted')
        f1 += f1_score(y_test1, y_pred, average='weighted')

    NO_OF_FOLDS = 10
    print "weighted results are"
    print "average precision is %f" %(p/NO_OF_FOLDS)
    print "average recall is %f" %(r/NO_OF_FOLDS)
    print "average f1 is %f" %(f1/NO_OF_FOLDS)


np.random.seed(42)
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('./datastories.twitter.300d.txt')
tweets = select_tweets()
gen_vocab()
X, y = gen_sequence()

MAX_SEQUENCE_LENGTH = max(map(lambda x:len(x), X))
print "max seq length is %d"%(MAX_SEQUENCE_LENGTH)
data = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
y = np.array(y)
y_user = y
data, y = sklearn.utils.shuffle(data, y)
X_user,X_tweet = getUserFeatures()
W = get_embedding_weights()
train_LSTM(data, y, X_user, y_user, EMBEDDING_DIM, W)
