import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
df_cancer.head()

X = df_cancer.drop(['target'], axis = 1)
y = df_cancer['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 20)

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_predict = svc_model.predict(X_test)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
sns.heatmap(confusion, annot=True)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
mlp = MLPClassifier(hidden_layer_sizes=(6,6,6,2), max_iter=2000)
mlp.fit(X_train,y_train)
y_predict = mlp.predict(X_test)

cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_cancer', 'is_healthy'],
                         columns=['predicted_cancer','predicted_healthy'])
sns.heatmap(confusion, annot=True)

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_poly.fit(X_train, y_train)
y_predict = svr_poly.predict(X_test)

import shap
explainer = shap.explainers.Permutation(svr_poly.predict, X_test)
shap_values_svc = explainer(X_test[:100], silent=True)
shap.plots.bar(shap_values_svc)

from sklearn.neural_network import MLPRegressor
mlp_regr = MLPRegressor(random_state=1, max_iter=2000).fit(X_train, y_train)
y_predict = mlp_regr.predict(X_test)

explainer = shap.explainers.Permutation(mlp_regr.predict, X_test)
shap_values_svc = explainer(X_test[:100], silent=True)
shap.plots.bar(shap_values_svc)