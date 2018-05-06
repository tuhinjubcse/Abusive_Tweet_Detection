import os,sys
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

file = '/Users/surabhibhargava/PycharmProjects/Nlp_project/tweet_text.csv'
train_file = 'train.csv'
test_file = 'test.csv'


f = pd.read_csv(file)
print(len(f))
header = ["tweet_id","text", "none", "racism", "sexism"]
X_train, X_test = train_test_split(f,test_size=0.1, random_state=42)
with open(train_file, 'w') as tf:
    x = csv.writer(tf)
    for index, row in X_train.iterrows():
        none, racism, sexism = 0, 0, 0
        if row["label"] == "none":
            none = 1
        if row["label"] == "racism":
            racism = 1
        if row["label"] == "sexism":
            sexism = 1
        x.writerow([row['tweet_id'], row['text'], none, racism, sexism])

with open(test_file, 'w') as tf:
    x = csv.writer(tf)
    for index, row in X_test.iterrows():
        none, racism, sexism = 0, 0, 0
        if row["label"] == "none":
            none = 1
        if row["label"] == "racism":
            racism = 1
        if row["label"] == "sexism":
            sexism = 1
        x.writerow([row['tweet_id'], row['text'], none, racism, sexism])

print(len(X_train))
