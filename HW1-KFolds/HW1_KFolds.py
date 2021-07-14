## Vincent Purcell
## Data Mining 2 - HW1 K Folds
## Some of code is modified from the code Professor Breitzman provided us.

import csv
import pandas as pd
import re
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,confusion_matrix

def is_empty_or_blank(msg):
    return re.search("^\s*$", msg)

with open('water_potability.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    columns = []
    data = []
    for row in csv_reader:
        if line_count == 0:
            data_series = pd.Series(row)
            features = list(data_series)
            columns = row
            line_count += 1
        else:
            data_series = pd.Series(row)
            data_row = list(data_series)
            result = any([is_empty_or_blank(elem) for elem in data_row])
            if result == False:
                data.append([float(i) for i in data_row])

data = numpy.array(data)
X = data[:,0:9]
Y = data[:,9]

# Columns and example data row
print(columns)
print(data[0])

kf = KFold(n_splits=5, random_state=35, shuffle=True)
kf

for index, (train, test) in enumerate(kf.split(X), 1):
    
    print(f'Split {index}:\n')
    
    print(test)
    print('\n')

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

for index, (train, test) in enumerate(kf.split(X), 1):
    
    print(f'Split {index}:\n')
    
    X_train = X[train]
    X_test = X[test]
    y_train = Y[train]
    y_test = Y[test]
    
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    mlp = MLPClassifier(hidden_layer_sizes=(30,30,30,3), max_iter=2000)
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    
    print('\n')