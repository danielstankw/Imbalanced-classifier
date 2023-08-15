import pickle

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, roc_curve
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.dummy import DummyClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
import plotly.express as px
from sklearn.metrics import auc


def plot_precision_vs_recall(prec, rec):
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Tradeoff')
    plt.grid()
    plt.show()


def plot_precision_recall_vs_threshold(prec, rec, thresh):
    f1 = 2 * ((prec * rec) / (prec + rec))
    plt.figure()
    plt.plot(thresh, prec[:-1], "b--", label='Precision')
    plt.plot(thresh, rec[:-1], "g-", label='Recall')
    plt.plot(thresh, f1[:-1], "k-", label='F1')
    plt.axvline(x=0.96, color='r', label='Decision threshold')
    plt.xlabel('Threshold')
    plt.title('Precision/ Recall/ F1 Score/ Threshold')
    plt.legend()
    plt.grid()
    plt.show()


def plot_interactive_ROC(y_train, y_pred_prob, string):
    fpr, tpr, thresh = roc_curve(y_train, y_pred_prob)
    fig = px.line(x=fpr, y=tpr, hover_name=thresh,
                  line_shape="spline", render_mode="svg", labels=dict(x='False Positive Rate', y='True Positive Rate'),
                  title=f'{string} ROC Curve (AUC={auc(fpr, tpr):.4f})', )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.show()

dff1 = pd.read_csv('circle_0.0008.csv')
dff2 = pd.read_csv('circle_0.0007.csv')
dff3 = pd.read_csv('circle_0.0006.csv')

# 3-no overlap, 2-overlap but not insert, 1-enough to insert
# we will combine 3 and 2 into 0 label
dff1['Case'] = dff1['Case'].replace([3, 2], 0)
dff2['Case'] = dff2['Case'].replace([3, 2], 0)
dff3['Case'] = dff3['Case'].replace([3, 2], 0)

t_cont1 = dff1['t_contact'][0]
t_cont2 = dff2['t_contact'][0]
t_cont3 = dff3['t_contact'][0]

df1 = dff1[dff1.time > t_cont1]
df2 = dff2[dff2.time > t_cont2]
df3 = dff3[dff3.time > t_cont1]

df = pd.concat([df1, df2, df3], ignore_index=True)

# ['time', 'Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Mx', 'My', 'Case', 't_contact']
feature_names = ['Fz', 'Mx', 'My']

X = df[feature_names]
y = df.Case

print(f"Ratio of 1's in training set {y.sum() / len(y) * 100}% and count is: {y.sum()}")
print(f"Ratio of 0's in training set {100 - y.sum() / len(y) * 100}% and count is: {len(y) - y.sum()}")

# Stratified sampling aims at splitting a data set so that each split is similar with respect to something.
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# we keep X_train/ test and y_train/test for the under sampled data
original_X_train, original_X_test, original_y_train, original_y_test = None, None, None, None

for train_index, test_index in sss.split(X, y):
    original_X_train, original_X_test = X.iloc[train_index], X.iloc[test_index]
    original_y_train, original_y_test = y.iloc[train_index], y.iloc[test_index]

# we set split_num=5 -> 25% for test, 75 train
print('Train length', len(original_X_train))
print('Test length', len(original_X_test))
print(f'{len(original_X_test) / len(original_X_train) * 100}% of the data is testing')
print()
print(f"Ratio of 1 in train {original_y_train.sum() / len(original_y_train) * 100} ")
print(f"Ratio of 1 in test {original_y_test.sum() / len(original_y_test) * 100}")

'''We have same distribution of labels in train and test data!'''

original_X_train = original_X_train.values
original_X_test = original_X_test.values
original_y_train = original_y_train.values
original_y_test = original_y_test.values

print('Dumb classifier accuracy', 1 - original_y_test.sum() / len(original_y_test))

print("Number of labels '1' in training set: {}".format(sum(original_y_train == 1)))
print("Number of labels '0' in training set: {} \n".format(sum(original_y_train == 0)))

sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(original_X_train, original_y_train)

print("After SMOTE, counts of label '1': {}".format(sum(y_train_res == 1)))
print("After SMOTE, counts of label '0': {} \n".format(sum(y_train_res == 0)))

# TODO: Dummy Classifier
clf = DummyClassifier(strategy='most_frequent').fit(X_train_res, y_train_res)
y_pred = clf.predict(original_X_test)
print('Accuracy Score : ' + str(accuracy_score(original_y_test, y_pred)))
print('Precision Score : ' + str(precision_score(original_y_test, y_pred)))
print('Recall Score : ' + str(recall_score(original_y_test, y_pred)))
print('F1 Score : ' + str(f1_score(original_y_test, y_pred)))
print('Confusion Matrix : \n' + str(confusion_matrix(original_y_test, y_pred)))
print(classification_report(original_y_test, y_pred))

# TODO: Grid Search for better models and better params
rdc_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2, 12, 2)),
              "min_samples_leaf": list(range(5, 10, 1))}

# TODO: focus on reducing false positives
# {'criterion': 'gini', 'max_depth': 10, 'min_samples_leaf': 8}
clf = GridSearchCV(RandomForestClassifier(), rdc_params, cv=5, verbose=5, n_jobs=3, scoring='average_precision')

# log_reg_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "penalty": ["l1", "l2"]}
# # select model using metric agnostic to te threshold and only after we tune the threshold
# clf = GridSearchCV(LogisticRegression(solver='liblinear'), log_reg_parameters, cv=5, verbose=5, n_jobs=3,
#                    scoring='f1') #average_precision

# clf.fit(X_train_res, y_train_res)
# print(clf.best_params_)
# print(clf.best_estimator_)

# log_best = clf.best_estimator_

# fit using resampled data
log_best = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_leaf=8)
log_best.fit(X_train_res, y_train_res)

# testing on TEST DATA
# make predictions on the original (un-resampled) data
y_test_predict = log_best.predict(original_X_test)
y_train_predict = log_best.predict(X_train_res)
# TODO: print confusion matrix
print('Perfect Confusion Matrix')
print(confusion_matrix(original_y_test, original_y_test))
print('Train Set Confusion Matrix')
print(confusion_matrix(y_train_res, y_train_predict))
print('Test Set Confusion Matrix')
print(confusion_matrix(original_y_test, y_test_predict))
# TODO: print classification report
print('Train Data - Balanced')
print(classification_report(y_train_res, y_train_predict))
print('Train Data - Unbalanced')
print(classification_report(original_y_test, y_test_predict))
# TODO: print ROC (standard)
# plot_roc_curve(log_best, X_train_res, y_train_res)
# plt.show()
# plot_roc_curve(log_best, original_X_test, y_test)
# plt.show()

# TODO: Obtaining probability predictions of TRAIN set
y_scores_train = log_best.predict_proba(X_train_res)[:, 1]
precisions_train, recalls_train, thresholds_train = precision_recall_curve(y_train_res, y_scores_train)

plot_interactive_ROC(y_train_res, y_scores_train, 'Train')
plot_precision_vs_recall(precisions_train, recalls_train)
plot_precision_recall_vs_threshold(precisions_train, recalls_train, thresholds_train)

# TODO: Obtaining probability predictions of TEST set
y_scores = log_best.predict_proba(original_X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(original_y_test, y_scores)

plot_interactive_ROC(original_y_test, y_scores, 'Test')
plot_precision_vs_recall(precisions, recalls)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#
# TODO Modifying Decision Boundary
threshold_93_precision = thresholds[np.argmax(precisions >= 0.96)]
print('threshold 93 precision', threshold_93_precision)

yscore = (y_scores >= threshold_93_precision)
print(classification_report(original_y_test, yscore))

# filename = 'RFClassifier.sav'
# pickle.dump(log_best, open(filename, 'wb'))
