import pickle
import time

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.dummy import DummyClassifier
import timeit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
data = np.array([1.0, 2.0, 3.0]).reshape(1, 3)

# (8838, 3)
loaded_model = pickle.load(open('RFClassifier.sav', 'rb'))
threshold_93_precision = 0.9355930
# we focus on positive class only
y_scores = loaded_model.predict_proba(data)[:, 1]
print(y_scores)
print(y_scores >= threshold_93_precision)
from hummingbird.ml import convert
from datetime import datetime
start = time.time()
y_scores = loaded_model.predict_proba(data)[:, 1]
print(time.time() - start)

