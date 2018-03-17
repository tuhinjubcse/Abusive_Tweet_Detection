from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS


def glove_tokenize(text):
    text = ''.join([c for c in text if c not in '!"#$%&()*+,-.:;=?@[\\]^_`{|}~\t\n'])
    words = text.split()
    words = [word for word in words if word not in STOPWORDS]
    return words
