import pandas as pd
import numpy as np
import os
from copy import deepcopy
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
import plotly.express as px
from sklearn.metrics import average_precision_score


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
    # plt.axvline(x=0.96, color='r', label='Decision threshold')
    plt.xlabel('Threshold')
    plt.title('Precision/ Recall/ F1 Score/ Threshold')
    plt.legend()
    plt.grid()
    plt.show()


def preprocess(df):
    precisions = df['precision'].values
    recalls = df['recall'].values
    thr_temp = df['threshold'].values
    lastElementIndex = len(thr_temp) - 1
    thresholds = thr_temp[:lastElementIndex]
    aps = df['aps'].values[0]
    return precisions, recalls, thresholds, aps

def pre_roc(df):
    fpr = df['fpr'].values
    tpr = df['tpr'].values
    auc = df['auc'].values[0]
    return fpr, tpr, auc

no_window = '/home/user/Desktop/LSTM/exp1_window_0/plot_data.csv'
window_3 = '/home/user/Desktop/LSTM/exp1_window_3/plot_data.csv'
window_6 = '/home/user/Desktop/LSTM/exp1_window_6/plot_data.csv'
window_9 = '/home/user/Desktop/LSTM/exp1_window_9/plot_data.csv'

df_no_window = pd.read_csv(no_window)
df_window3 = pd.read_csv(window_3)
df_window6 = pd.read_csv(window_6)
df_window9 = pd.read_csv(window_9)


p0, r0, t0, aps0 = preprocess(df_no_window)
p3, r3, t3, aps3 = preprocess(df_window3)
p6, r6, t6, aps6 = preprocess(df_window6)
p9, r9, t9, aps9 = preprocess(df_window9)

# PR Curve
plt.figure()
plt.plot(r0, p0, 'r', label=f'PR of no window, area {np.round(aps0, 3)}')
plt.plot(r3, p3, '--b', label=f'PR of window: 3, area {np.round(aps3, 3)}')
plt.plot(r6, p6, 'k', label=f'PR of window: 6 area {np.round(aps6, 3)}')
plt.plot(r9, p9, '--g', label=f'PR of window: 9, area {np.round(aps9, 3)}')
plt.legend()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.grid()
# plt.show()

no_window_roc = '/home/user/Desktop/LSTM/exp1_window_0/plot_data2.csv'
window_3_roc = '/home/user/Desktop/LSTM/exp1_window_3/plot_data2.csv'
window_6_roc = '/home/user/Desktop/LSTM/exp1_window_6/plot_data2.csv'
window_9_roc = '/home/user/Desktop/LSTM/exp1_window_9/plot_data2.csv'

df_no_window_roc = pd.read_csv(no_window_roc)
df_window3_roc = pd.read_csv(window_3_roc)
df_window6_roc = pd.read_csv(window_6_roc)
df_window9_roc = pd.read_csv(window_9_roc)

fpr0, tpr0, auc0 = pre_roc(df_no_window_roc)
fpr3, tpr3, auc3 = pre_roc(df_window3_roc)
fpr6, tpr6, auc6 = pre_roc(df_window6_roc)
fpr9, tpr9, auc9 = pre_roc(df_window9_roc)


plt.figure()
plt.plot(fpr0, tpr0, 'r', label="No window |AUC=" + str(np.round(auc0, 3)))
plt.plot(fpr3, tpr3, '--b', label="Window size:3 |AUC=" + str(np.round(auc3, 3)))
plt.plot(fpr6, tpr6, 'k', label="Window size:6 |AUC=" + str(np.round(auc6, 3)))
plt.plot(fpr9, tpr9, '--g', label="Window size:9 |AUC=" + str(np.round(auc9, 3)))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve')
plt.grid()
plt.legend(loc=4)
plt.show()


# plot_precision_vs_recall(precisions, recalls)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plot_roc(fpr, tpr, auc)
