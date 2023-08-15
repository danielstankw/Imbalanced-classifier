import os
import pandas as pd
import numpy as np
from copy import deepcopy

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score, f1_score
import plotly.express as px
from sklearn.metrics import average_precision_score

keras.backend.clear_session()


def plot_roc(fpr, tpr, auc):
    """Plots ROC curve"""
    plt.figure()
    plt.title('ROC')
    plt.plot(fpr, tpr, label="AUC=" + str(np.round(auc, 3)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend()
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend(loc=4)
    plt.grid()
    plt.show()


def plot_precision_vs_recall(prec, rec, aps):
    plt.figure()
    plt.plot(rec, prec, label=f'PR of window 12, area {np.round(aps, 3)}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.legend()
    plt.grid()
    plt.show()


def plot_loss(history, label, col):
    plt.semilogy(history.epoch, history.history['loss'], color=col, label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'], col, label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def plot_roc_interactive(name, y_train, y_pred_prob):
    fpr, tpr, thresh = roc_curve(y_train, y_pred_prob)
    fig = px.line(x=fpr, y=tpr, hover_name=thresh,
                  line_shape="spline", render_mode="svg", labels=dict(x='False Positive Rate', y='True Positive Rate'),
                  title=f'{name} ROC Curve')
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.show()


def f1(precision, recall):
    # print((precision + recall))
    return 2 * (precision * recall) / (precision + recall)


def plot_precision_recall_vs_threshold(prec, rec, thresh):
    f1_vec = f1(prec, rec)

    plt.figure()
    plt.plot(thresh, prec[:-1], "b--", label='Precision')
    plt.plot(thresh, rec[:-1], "g-", label='Recall')
    plt.plot(thresh, f1_vec[:-1], "k-", label='F1')
    # plt.axvline(x=0.96, color='r', label='Decision threshold')
    plt.xlabel('Threshold')
    plt.title('Precision/ Recall/ F1 Score/ Threshold')
    plt.legend()
    plt.grid()
    plt.show()


def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(2, 2, n + 1)
        plt.plot(history.epoch, history.history[metric], label='Train')
        plt.plot(history.epoch, history.history['val_' + metric], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()

# TODO: 6,7,8
# dff1 = pd.read_csv('circle2_0.001.csv')
# dff2 = pd.read_csv('circle2_0.00095.csv')
# dff3 = pd.read_csv('circle2_0.0009.csv')
# dff4 = pd.read_csv('circle2_0.00085.csv')
dff5 = pd.read_csv('circle2_0.0008.csv')
# dff6 = pd.read_csv('circle2_0.00075.csv')
dff7 = pd.read_csv('circle2_0.0007.csv')
# dff8 = pd.read_csv('circle2_0.00065.csv')
dff9 = pd.read_csv('circle2_0.0006.csv')

# 3-no overlap, 2-overlap but not insert, 1-enough to insert
# we will combine 3 and 2 into 0 label
dffs = [dff5, dff7, dff9]

df = pd.DataFrame()

for dff in dffs:
    dff['Case'] = dff['Case'].replace([3, 2], 0)
    t_cont = dff['t_contact'][0]
    dff = dff[dff.time > t_cont]
    df = pd.concat([df, dff], ignore_index=True)

# file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection/all_with_label.csv"
# df = pd.read_csv(file_name)

SAVE = False
LOAD = False
folder_name = '16_window_12_model_3'
path_name = 'Weights_folder_1/Weights'
save_data = False
save_weights = False
load_weights = False

window_len = 12
use_smote = False
plot = True

EPOCHS = 50  # 100
BATCH_SIZE = 1024  # 2048

# ['time', 'Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Mx', 'My', 'Case', 't_contact']
feature_names = ['Fz', 'Mx', 'My']
X = df[feature_names].to_numpy()
y = df.Case.to_numpy()

if window_len:
    n_features = len(feature_names)  # 3

    row = X.shape[0] + 1 - window_len
    col_len = n_features * window_len
    new_x = np.zeros((row, col_len))
    new_y = np.zeros((row, 1))

    for i in range(len(new_x)):
        new_x[i] = X[i:i + window_len].reshape(1, col_len)[0][::-1]
        new_y[i] = y[i + (window_len - 1)]

    y = deepcopy(new_y)
    X = deepcopy(new_x)

print(f"Ratio of 1's in training set {y.sum() / len(y) * 100}% and count is: {y.sum()}")
print(f"Ratio of 0's in training set {100 - y.sum() / len(y) * 100}% and count is: {len(y) - y.sum()}")

# Stratified sampling aims at splitting a data set so that each split is similar with respect to something.
# sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# we keep X_train/ test and y_train/test for the under sampled data
original_X_train, original_X_test, original_y_train, original_y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

total = len(original_y_train)
pos = original_y_train.sum()
neg = total - pos

# we set split_num=5 -> 25% for test, 75 train
print('Train length', len(original_X_train))
print('Test length', len(original_X_test))
print(f'{len(original_X_test) / len(original_X_train) * 100}% of the data is testing')
print()
print(f"Ratio of 1 in train {original_y_train.sum() / len(original_y_train) * 100} ")
print(f"Ratio of 1 in test {original_y_test.sum() / len(original_y_test) * 100}")

'''We have same distribution of labels in train and test data!'''

print('Dumb classifier accuracy', (1 - original_y_test.sum() / len(original_y_test)) * 100)

print(f'Train set size {len(original_y_train)}')
print("Number of labels '1' in training set: {}".format(original_y_train.sum()))
print("Number of labels '0' in training set: {} \n".format(len(original_y_test) - original_y_train.sum()))

if use_smote:
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(original_X_train, original_y_train)

    original_X_train = deepcopy(X_train_res)
    original_y_train = deepcopy(y_train_res)
    print("After SMOTE, counts of label '1': {}".format(original_y_train.sum()))
    print("After SMOTE, counts of label '0': {} \n".format(len(original_y_test) - original_y_train.sum()))

tf.random.set_seed(42)

n_inputs = original_X_train.shape[1]  # (77862, 3)

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
]


def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    nn = keras.Sequential()
    nn.add(keras.layers.Dense(16, input_shape=(n_inputs,), activation='relu'))
    # nn.add(keras.layers.Dense(32, activation='relu'))
    # nn.add(keras.layers.Dropout(0.5))
    # nn.add(keras.layers.Dense(64, activation='relu'))
    nn.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))

    nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=metrics)
    return nn


if save_weights:
    model = make_model()
    model.save_weights(path_name)
    print('Weights and bias has been saved!')

new_path = os.path.join('/home/user/Desktop/LSTM/', folder_name)
if not os.path.exists(new_path):
    os.mkdir(new_path)

model = make_model()
if load_weights:
    model.load_weights(path_name)

if LOAD:
    model = keras.models.load_model("final_model_3")

model_orig_history = model.fit(
    original_X_train,
    original_y_train,
    epochs=EPOCHS,
    validation_data=(original_X_test, original_y_test),
    shuffle=True)

# TODO: use plotter.py

if SAVE:
    model.save('final_model_3')

# plot_loss(model_orig_history, "Original Trained Model Loss", col='red')
# plt.show()
# plot_metrics(model_orig_history)
# plt.show()

train_pred_ori = model.predict(original_X_train, batch_size=200)
test_pred_ori = model.predict(original_X_test, batch_size=200)

print('Classification Report for: Original TRAIN SET')
print(classification_report(original_y_train, train_pred_ori > 0.5))
print('Classification Report for: Original TEST SET')
print(classification_report(original_y_test, test_pred_ori > 0.5))

# plot_roc_interactive('Train', original_y_train, train_pred_ori)
# plot_roc_interactive('Test', y_test, test_pred_ori)
#
fpr, tpr, _ = roc_curve(original_y_test, test_pred_ori)
auc = roc_auc_score(original_y_test, test_pred_ori)
auc_vec = auc * np.ones(len(fpr))
# plot_roc(fpr, tpr, auc)

save_roc = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'auc': auc_vec})
new_path3 = os.path.join(new_path, 'plot_data2.csv')
if save_data:
    save_roc.to_csv(new_path3)

aps = average_precision_score(original_y_test, test_pred_ori)
precisions, recalls, thresholds = precision_recall_curve(original_y_test, test_pred_ori)
# plot_precision_vs_recall(precisions, recalls, aps)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

aps_vec = aps * np.ones(len(precisions))
# due to size constraints
thresh = deepcopy(thresholds)
thresh = np.append(thresh, thresh[-1])
save_df = pd.DataFrame({'aps': aps_vec, 'precision': precisions, 'recall': recalls, 'threshold': thresh})
new_path2 = os.path.join(new_path, 'plot_data.csv')
if save_data:
    save_df.to_csv(new_path2)

if plot:
    plot_precision_vs_recall(precisions, recalls, aps)
    plot_roc(fpr, tpr, auc)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

# find the index of the threshold that yields the highest F1 score
best_index = np.argmax([f1(precisions[i], recalls[i]) for i in range(len(thresholds))])
f1_vec = f1(precisions, recalls)
# find the threshold value that yields the highest F1 score
print()
print(f"Threshold that yields max F1 score of:{f1_vec[best_index]} is {thresholds[best_index]}")
print(f"The precision is: {precisions[best_index]} and recall: {recalls[best_index]} at this threshold")
print()

prec_th = 0.95
threshold_precision = thresholds[np.argmax(precisions >= prec_th)]
print(f'The {prec_th * 100} precision threshold is: {threshold_precision}')

yscore = (test_pred_ori >= threshold_precision)
print(classification_report(original_y_test, yscore))

# if SAVE:
#     model_res.save('my_model_custom_bias')
# if LOAD:
#     model_orig = keras.models.load_model("my_model_custom_bias")

# train_pred_ori = model_orig.predict(original_X_train, batch_size=200)
# test_pred_ori = model_orig.predict(original_X_test, batch_size=200)
#
# print('Classification Report for: Original TRAIN SET')
# print(classification_report(original_y_train, train_pred_ori > 0.5))
# print('Classification Report for: Original TEST SET')
# print(classification_report(y_test, test_pred_ori > 0.5))
# #
# # plot_roc("Train Baseline", original_y_train, train_pred_ori, color='blue')
# # plot_roc("Test Baseline", y_test, test_pred_ori, color='blue', linestyle='--')
# # plt.legend(loc='lower right')
# # plt.show()
# # TODO: for tuning
# # plot_roc_interactive('Train', original_y_train, train_pred_ori)
# # plot_roc_interactive('Test', y_test, test_pred_ori)
#
#
# y_scores = model_orig.predict(original_X_test)
# precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#
# # TODO Modifying Decision Boundary
# threshold_93_precision = thresholds[np.argmax(precisions >= 0.84)]
# print('threshold 93 precision', threshold_93_precision)
#
# yscore = (y_scores >= threshold_93_precision)
# print(classification_report(y_test, yscore))
#
#
#
#
