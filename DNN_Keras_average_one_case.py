import csv
import os
import pandas as pd
import numpy as np
from copy import deepcopy

from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score, f1_score
import plotly.express as px
from sklearn.metrics import average_precision_score

SAVE = False
LOAD = False

keras.backend.clear_session()


def plot_roc(fpr, tpr, auc):
    plt.plot(fpr, tpr, label="AUC=" + str(np.round(auc, 3)))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


def plot_precision_vs_recall(prec, rec, aps):
    plt.figure()
    plt.plot(rec, prec, label=f'PR of no window, area {np.round(aps, 3)}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Tradeoff')
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

def plot_loss(history, label, col):
    plt.semilogy(history.epoch, history.history['loss'], color=col, label='Train ' + label)
    plt.semilogy(history.epoch, history.history['val_loss'], col, label='Val ' + label,
                 linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

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


# dff1 = pd.read_csv('circle_0.0008.csv')
# dff2 = pd.read_csv('circle_0.0007.csv')
# dff3 = pd.read_csv('circle_0.0006.csv')
#
dff1 = pd.read_csv('circle2_0.0008.csv')
dff2 = pd.read_csv('circle2_0.0007.csv')
dff3 = pd.read_csv('circle2_0.0006.csv')

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

folder_name = 'exp2_window_0'
path_name = 'Weights_folder_0/Weights'
# window_len = 24
use_smote = False
shuffle = True

save_data = False
save_weights = False
load_weights = False

n_models = 30
EPOCHS = 50  # 100
BATCH_SIZE = 1024  # 2048

# Index(['time', 'x', 'y', 'z', 'rx', 'ry', 'rz', 'Fx', 'Fy', 'Fz', 'Vx', 'Vy',
#        'Vz', 'Mx', 'My', 'Mz', 'Case', 't_contact'],
windows = [0, 3, 6, 9, 12, 16, 18, 24]
for window_len in windows:
    data_df = {}
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
    sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

    # we keep X_train/ test and y_train/test for the under sampled data
    original_X_train, original_X_test, original_y_train, original_y_test = None, None, None, None

    for train_index, test_index in sss.split(X, y):
        original_X_train, original_X_test = X[train_index], X[test_index]
        original_y_train, original_y_test = y[train_index], y[test_index]

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
        nn.add(keras.layers.Dense(32, activation='relu'))
        # nn.add(keras.layers.Dropout(0.5))
        # nn.add(keras.layers.Dense(64, activation='relu'))
        nn.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))

        nn.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='binary_crossentropy',
            metrics=metrics)
        return nn


    # the modified bias is much smaller than one with naive initialization, which speeds up learning!
    # saving initial weights for analysis: model with modified bias

    if save_weights:
        model = make_model()
        model.save_weights(path_name)
        print('Weights and bias has been saved!')

    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    # new_path = os.path.join('/home/user/Desktop/LSTM/', folder_name)
    # if not os.path.exists(new_path):
    #     os.mkdir(new_path)

    fpr_vec = []
    tpr_vec = []
    auc_vec = []
    prec_vec = []
    rec_vec = []
    aps_vec = []

    f1_weighted = []
    f1_default = []
    f1_micro = []
    print(original_X_train.shape)

    for i in range(n_models):
        keras.backend.clear_session()

        model = make_model()
        if load_weights:
            model.load_weights(path_name)
        model_orig_history = model.fit(
            original_X_train,
            original_y_train,
            epochs=EPOCHS,
            validation_split=0.2,
            shuffle=shuffle)
            # class_weight=class_weight)

        # plot_loss(model_orig_history, "Original Trained Model Loss", col='red')
        # plt.show()

        train_pred_ori = model.predict(original_X_train, batch_size=200)
        test_pred_ori = model.predict(original_X_test, batch_size=200)

        fpr, tpr, _ = roc_curve(original_y_test, test_pred_ori)
        fpr_vec.append(fpr)
        tpr_vec.append(tpr)

        auc = roc_auc_score(original_y_test, test_pred_ori)
        auc_vec.append(auc)

        aps = average_precision_score(original_y_test, test_pred_ori)
        aps_vec.append(aps)
        precisions, recalls, thresholds = precision_recall_curve(original_y_test, test_pred_ori)
        prec_vec.append(precisions)
        rec_vec.append(recalls)

        temp = test_pred_ori > 0.5
        f1_default.append(f1_score(original_y_test, temp))
        print(f1_score(original_y_test, temp))
        f1_weighted.append(f1_score(original_y_test, temp, average='weighted'))
        f1_micro.append(f1_score(original_y_test, temp, average='micro'))
        i += 1

    data_df['aps'] = aps_vec
    data_df['auc'] = auc_vec
    data_df['f1_def'] = f1_default
    data_df['f1_weight'] = f1_weighted
    data_df['f1_micro'] = f1_micro
    df_save = pd.DataFrame(data_df)
    name = "Run_16_32_30models_window" + str(window_len) + ".csv"
    df_save.to_csv(name, index=False)



# auc vector
# aps curve
# keys = ['AUROC', 'AUPRC']
# values = [auc_vec, aps_vec]
# my_dict = dict(zip(keys, values))
#
# # Open a file for writing
# with open('16_32_window12_results.csv', 'w', newline='') as file:
#     # Create a fieldnames list
#     fieldnames = keys
#
#     # Create a csv.DictWriter object
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#
#     # Write the header row
#     writer.writeheader()
#
#     # Write the data rows
#     writer.writerows([my_dict])

#
# # TODO: ROC
# tprs = []
# colors = ['b', 'b', 'b', 'b', 'b']
# mean_fpr = np.linspace(0, 1, 101)
# plt.figure()
# for i in range(n_models):
#     plt.plot(fpr_vec[i], tpr_vec[i], alpha=0.3, label=f"AUC=" + str(np.round(auc_vec[i], 3)))
#     interp_tpr = np.interp(mean_fpr, fpr_vec[i], tpr_vec[i])
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     i += 1
#
# tprs = np.array(tprs)
# mean_tpr = tprs.mean(axis=0)
# mean_auc = np.mean(auc_vec)
# std_auc = np.std(auc_vec)
#
# plt.plot(mean_fpr, mean_tpr, color='b',
#          label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc), lw=2, alpha=0.8)
#
# std_tpr = tprs.std(axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label=r"$\pm$ 1 std. dev.")
# plt.plot([0, 1], [0, 1], 'r--')
# plt.legend()
# plt.xlim([-0.01, 1.01])
# plt.ylim([-0.01, 1.01])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title(f'ROC for window of size {window_len}')
# plt.grid()
#
# # TODO Precision-Recall
# # x - recall, y - precision
#
# plt.figure()
# for i in range(n_models):
#     plt.plot(rec_vec[i], prec_vec[i], alpha=0.7, label=f'{window_len} window | Area: {np.round(aps_vec[i], 3)}')
#     i += 1
#
# plt.xlim([-0.01, 1.01])
# plt.ylim([-0.01, 1.01])
# plt.legend()
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title(f'Precision Recall Curve: Mean APS {np.round(np.mean(aps_vec), 3)} std: {np.round(np.std(aps_vec), 3)} for window of size {window_len}')
# plt.grid()
# plt.show()
#
#
# def average(lst):
#     return sum(lst) / len(lst)
#
#
# print('F1 Default avg', average(f1_default))
# print('F1 Default', f1_default)
# print('F1 Weighted avg', average(f1_weighted))
# print('F1 Weighted ', f1_weighted)
# print('F1 Micro avg', average(f1_micro))
# print('F1 Micro', f1_micro)
