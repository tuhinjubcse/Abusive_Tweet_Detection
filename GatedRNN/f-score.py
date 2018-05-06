import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

test_file = 'test.csv'
output = 'output.csv'

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
output = pd.read_csv('output.csv')

labels = []
predictions = []

for index, row in output.iterrows():
    max_val = max(float(row['none']), float(row['sexism']), float(row['racism']))
    if float(row['none']) == max_val:
        predictions.append(0)
    if float(row['sexism']) == max_val:
        predictions.append(2)
    if float(row['racism']) == max_val:
        predictions.append(1)

for index, row in test.iterrows():
    if int(row['none']) == 1:
        labels.append(0)
    if int(row['racism']) == 1:
        labels.append(1)
    if int(row['sexism']) == 1:
        labels.append(2)
target_names = ["none", "racism", "sexism"]
f = f1_score(labels, predictions, average='weighted')
print(f)

for i in range(len(predictions)):
    if predictions[i] != 0 and labels[i] == 0 and predictions[i]!= labels[i]:
        print(i,print(test['text'][i]))
        # print(i)

print(classification_report(labels, predictions, target_names=target_names))

# print(predictions)
# print(labels)