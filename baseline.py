import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score,classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from collections import defaultdict
from data_handler import get_data
from my_tokenizer import glove_tokenize
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
import sklearn
from scipy.sparse import coo_matrix, hstack,vstack


def get_top_features(tfidf_transformer):
    features_by_gram = defaultdict(list)
    for f, w in zip(tfidf_transformer.get_feature_names(), tfidf_transformer.idf_):
        features_by_gram[len(f.split(' '))].append((f, w))
    top_n = 100
    for gram, features in features_by_gram.items():
        top_features = sorted(features, key=lambda x: x[1], reverse=True)[:top_n]
        top_features = [f[0] for f in top_features]
        print('{}-gram top:'.format(gram), top_features)
        if 'mkr' in top_features:
            print 'Yessss'

def getAbusiveFeatures():
    f = open('abusive_dict.txt','r')
    m = {}
    for line in f:
        line = line.strip()
        m[line]=True
    tweets = get_data()
    X = []
    for tweet in tweets:
        text = glove_tokenize(tweet['text'].lower())
        c = 0
        for word in text:
            if word in m:
                c = c+1
        X.append(c)
    return np.array(X)


def get_tfidf_features():
    tweets = get_data()
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
        if y_map[tweet['label']]==2:
            X.append(text)
            y.append(y_map[tweet['label']])
    tfidf_transformer = TfidfVectorizer(ngram_range=(1,1), analyzer='word',stop_words='english',max_features=10)
    X_tfidf = tfidf_transformer.fit_transform(X)
    print(X_tfidf.shape)

    get_top_features(tfidf_transformer)

    return X_tfidf, np.array(y)


def train_MLP(train_X, train_Y, test_X, test_Y, hidden_layer_sizes):
    print("Training Multi-layer perceptron " + str(tuple))
    classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes)
    classifier.fit(train_X, train_Y)

    pred_Y = classifier.predict(test_X)
    print(confusion_matrix(test_Y, pred_Y))

    print("Accuracy = " + str(classifier.score(test_X, test_Y)))
    print("Macro-f1 = " + str(f1_score(test_Y, pred_Y, average="macro")))
    print("Weighted-f1 = " + str(f1_score(test_Y, pred_Y, average="weighted")))


def train_randomforest(train_X, train_Y, test_X, test_Y, estimators,iter):
    print("Training random_forest " + str(estimators))
    # class_weights = {}
    # class_weights[0] = len([y for y in train_Y if y==0])
    # class_weights[1] = len([y for y in train_Y if y==1])
    # class_weights[2] = len([y for y in train_Y if y==2])
    classifier = RandomForestClassifier(n_estimators=estimators,class_weight = None)
    classifier.fit(train_X, train_Y)

    f = open('./baseline/cv_'+str(iter),'w')
    pred_Y = classifier.predict(test_X)
    for i in range(len(pred_Y)):
        f.write(str(pred_Y[i])+'\n')

    f = open('./test/cv_'+str(iter),'w')
    for i in range(len(test_Y)):
        f.write(str(test_Y[i])+'\n')

    # print(confusion_matrix(test_Y, pred_Y))

    # accuracy = classifier.score(test_X, test_Y)
    print classification_report(test_Y, pred_Y)
    macro_precision = precision_score(test_Y, pred_Y, average="macro")
    macro_recall = recall_score(test_Y, pred_Y, average="macro")
    macro_f1 = f1_score(test_Y, pred_Y, average="macro")

    weighted_precision = precision_score(test_Y, pred_Y, average="weighted")
    weighted_recall = recall_score(test_Y, pred_Y, average="weighted")
    weighted_f1 = f1_score(test_Y, pred_Y, average="weighted")

    # print("Accuracy = " + str(accuracy))
    # print("Macro-precision = " + str(macro_precision))
    # print("Macro-recall = " + str(macro_recall))
    # print("Macro-f1 = " + str(macro_f1))

    # print("Weighted-precision = " + str(weighted_precision))
    # print("Weighted-recall = " + str(weighted_recall))
    # print("Weighted-f1 = " + str(weighted_f1))

    return macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1


def train_GBC(train_X, train_Y, test_X, test_Y, estimators):
    accuracy = 0
    macro_f1 = 0
    weighted_f1 = 0
    print("Training GBC " + str(estimators))
    for i in range(1):
        print(i)
        classifier = GradientBoostingClassifier(n_estimators=estimators)
        classifier.fit(train_X, train_Y)

        pred_Y = classifier.predict(test_X)
        # print(confusion_matrix(test_Y, pred_Y))

        accuracy += classifier.score(test_X, test_Y)
        macro_f1 += f1_score(test_Y, pred_Y, average="macro")
        weighted_f1 += f1_score(test_Y, pred_Y, average="weighted")

    print("Accuracy = " + str(accuracy / 1))
    print("Macro-f1 = " + str(macro_f1 / 1))
    print("Weighted-f1 = " + str(weighted_f1 / 1))


def train_SVM(train_X, train_Y, test_X, test_Y):
    print("Training SVM")
    C_range = np.logspace(2, 2, 1, base=2)
    gamma_range = np.logspace(-2, -2, 1, base=2)
    param_grid = dict(gamma=gamma_range, C=C_range)

    svc = SVC(decision_function_shape='ovo')
    classifier = GridSearchCV(svc, param_grid=param_grid)

    classifier.fit(train_X, train_Y)

    print("Best params = " + str(classifier.best_params_))

    pred_Y = classifier.predict(test_X)
    print(confusion_matrix(test_Y, pred_Y))

    print("Accuracy = " + str(classifier.score(test_X, test_Y)))
    print("Macro-f1 = " + str(f1_score(test_Y, pred_Y, average="macro")))
    print("Weighted-f1 = " + str(f1_score(test_Y, pred_Y, average="weighted")))


if __name__ == '__main__':
    np.random.seed(42)
    # Xabs = getAbusiveFeatures()
    X, Y = get_tfidf_features()
    # X = np.concatenate((X, Xabs), axis=1)
    # print X[0]
    # X, Y = sklearn.utils.shuffle(X, Y)
    # print X[12] , type(X[12]), X[12].shape
    # # X = np.asarray(X)
    # # Y = np.asarray(Y)
   
    # prec_w, recall_w, f1_w, prec_m, recall_m, f1_m = 0., 0., 0., 0., 0., 0.
    # c = 1
    # cv_object = KFold(n_splits=10, shuffle=True, random_state=42)
    # for train_index, test_index in cv_object.split(X):
    #     X_train, Y_train = X[train_index], Y[train_index]
    #     X_test, Y_test = X[test_index], Y[test_index]

    #     macro_precision, macro_recall, macro_f1, weighted_precision, weighted_recall, weighted_f1 = train_randomforest(
    #         X_train, Y_train, X_test, Y_test, 500,c)

    #     prec_w += weighted_precision
    #     prec_m += macro_precision
    #     recall_w += weighted_recall
    #     recall_m += macro_recall
    #     f1_w += weighted_f1
    #     f1_m += macro_f1
    #     c = c+1

    # print("weighted results are")
    # print("average precision is %f" % (prec_w / 10))
    # print("average recall is %f" % (recall_w / 10))
    # print("average f1 is %f" % (f1_w / 10))

    # print("macro results are")
    # print("average precision is %f" % (prec_m / 10))
    # print("average recall is %f" % (recall_m / 10))
    # print("average f1 is %f" % (f1_m / 10))

# train_GBC(train_X_tfidf.todense(), train_Y, dev_X_tfidf.todense(), dev_Y, 500)
# train_SVM(train_X_tfidf.todense(), train_Y, dev_X_tfidf.todense(), dev_Y)
# train_MLP(train_X_tfidf.todense(), train_Y, dev_X_tfidf.todense(), dev_Y, (100,))





