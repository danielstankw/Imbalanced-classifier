from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve, roc_auc_score, f1_score
from sklearn.metrics import average_precision_score
import pandas as pd
import numpy as np
from copy import deepcopy
import seaborn as sns
from sklearn.metrics import classification_report,average_precision_score, precision_recall_curve, roc_curve, roc_auc_score, f1_score, confusion_matrix, auc
from sklearn.model_selection import GroupKFold, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report,average_precision_score, precision_recall_curve, roc_curve, roc_auc_score, f1_score, confusion_matrix, auc, precision_score, recall_score


def plot_prc_and_roc(y_true, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    average_precision = average_precision_score(y_true, y_pred)

    plt.figure()
    plt.step(recalls, precisions, color='b', alpha=1.0, where='post')
    plt.fill_between(recalls, precisions, step='post', alpha=0.1, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.grid()

    plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
        average_precision))
    plt.show()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    areaUnderROC = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic: \
            Area under the curve = {0:0.2f}'.format(areaUnderROC))
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def plot_precision_recall_vs_threshold(y_true, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_vec = 2 * (precisions * recalls) / (precisions + recalls)
    plt.figure()
    plt.plot(thresholds, precisions[:-1], "b--", label='Precision')
    plt.plot(thresholds, recalls[:-1], "g-", label='Recall')
    plt.plot(thresholds, f1_vec[:-1], "k-", label='F1')
    plt.xlabel('Threshold')
    plt.title('Precision/ Recall/ F1 Score/ Threshold')
    plt.legend()
    plt.grid()
    plt.show()


def dnn_stratified_kfold_cross_val(X, y, epoch, batch_size, callback, print_info, metrics):
    """
    StratifiedKFold cross-validation.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=False)
    aucpr_scores = []
    roc_scores = []
    prec_scores = []
    recall_scores = []

    for fold_id, (train_index, valid_index) in enumerate(skf.split(X, y)):
        print(f"Fold {fold_id}:")
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        if print_info:
            print(f"Ratio of 1's in train set {np.round((y_train.sum()) / len(y_train), 4) * 100}%")
            print(f"Ratio of 1's in validation set {np.round((y_valid.sum() / len(y_valid)), 4) * 100}%")

        n_inputs = X_train.shape[1]
        # make model
        model = make_model(n_inputs, metrics=metrics)
        # fit model
        model_history = model.fit(X_train, y_train,
                                  validation_data=(X_valid, y_valid),
                                  epochs=epoch,
                                  batch_size=batch_size,
                                  verbose=0,  # 1
                                  callbacks=[callback],
                                  shuffle=True)

        y_val_pred = model.predict(X_valid)
        roc = roc_auc_score(y_valid, y_val_pred)
        aucpr = average_precision_score(y_valid, y_val_pred)
        prec = precision_score(y_valid, y_val_pred > 0.5)
        recall = recall_score(y_valid, y_val_pred > 0.5)
        if print_info:
            print('--------------------------------------')
            print(f"Aucpr score for fold {fold_id} is : {aucpr}")
            print(f"ROC score for fold {fold_id} is : {roc}")
            print(f"Precision  : {prec}")
            print(f"Recall : {recall}")
            print('--------------------------------------')

        aucpr_scores.append(aucpr)
        roc_scores.append(roc)
        prec_scores.append(prec)
        recall_scores.append(recall)

    print()
    print(
        f"Mean across all folds:\nAUCPR: {np.mean(aucpr_scores)}\nROC {np.mean(roc_scores)}\nPrecision: {np.mean(prec_scores)}\nRecall: {np.mean(recall_scores)}")


def get_scores_thresholds(X_test, y_test, model, algo=None):
    thresholds = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    for thresh in thresholds:
        if algo == "XGB":
            y_pred = model.predict_proba(X_test)[:, 1]
        else:
            y_pred = model.predict(X_test)

        print('threshold:', thresh)
        print(confusion_matrix(y_test, y_pred > thresh))
        print(classification_report(y_test, y_pred > thresh))
        print('--------------------------------------')


def make_model(n_inputs, metrics=None, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    nn = keras.Sequential()
    nn.add(keras.layers.Dense(16, input_shape=(n_inputs,), activation='relu'))
    nn.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))
    # testing to reduce overfit
    #     nn.add(keras.layers.Dropout(0.1))

    nn.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
               loss='binary_crossentropy',
               metrics=metrics)
    # metrics simply tell us what we will be able to see in the log and on plot
    # they do are NOT used for optimization!

    return nn

df = pd.read_csv('all_data.csv')

df = df.drop(['Unnamed: 0.1','Unnamed: 0','t'], axis=1)

feature_names = ['Fz', 'Mx', 'My']
X_df = df[feature_names]
X = X_df.to_numpy()
y = df.Case.to_numpy()

window_len = 100
#
# if window_len:
#     n_features = len(feature_names)  # 3
#
#     row = X.shape[0] + 1 - window_len
#     col_len = n_features * window_len
#     new_x = np.zeros((row, col_len))
#     new_y = np.zeros((row, 1))
#
#     for i in range(len(new_x)):
#         new_x[i] = X[i:i + window_len].reshape(1, col_len)[0][::-1]
#         new_y[i] = y[i + (window_len - 1)]
#
#     y = deepcopy(new_y)
#     X = deepcopy(new_x)

if window_len:
    n_features = len(feature_names)  # 3

    row = X.shape[0] + 1 - window_len
    col_len = n_features * window_len
    new_x = np.zeros((row, col_len))
    new_y = np.zeros((row, 1))
    X_strided = np.lib.stride_tricks.as_strided(X, shape=(row, window_len, n_features), strides=(X.strides[0], X.strides[0], X.strides[1]))
    new_x = X_strided.reshape(row,col_len)[:,::-1]
    new_y = y[window_len-1:]

    y = new_y
    X = new_x


print()
# total = len(y)
# pos = int(y.sum())
# neg = total - pos
# weight = neg/pos # 13.5
#
# print(f"Total Dataset size: {X.shape}")
# print(f"Ratio of 1's in dataset {(pos/total) * 100}% and count is: {pos}")
# print(f"Ratio of 0's in dataset {(neg/total) * 100}% and count is: {neg}")
#
# METRICS = [
#     # keras.metrics.TruePositives(name='tp'),
#     # keras.metrics.FalsePositives(name='fp'),
#     # keras.metrics.TrueNegatives(name='tn'),
#     # keras.metrics.FalseNegatives(name='fn'),
#     # keras.metrics.Precision(name='precision'),
#     keras.metrics.Recall(name='recall'),
#     keras.metrics.AUC(name='auc'),
#     keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
# ]
#
# n_inputs = X.shape[1]
# print('Input shape',n_inputs)
# # define params
#
# #https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_prc', # validation precision
#     mode='max', # specify whether we seek to min/max given metrics
#     verbose=1,  # print training epoch on which we stopped
#     patience=20, #after how many epochs of no improvement do we stop
#     restore_best_weights=True)
#
# # create and compile network
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)
# n_inputs = X_train.shape[1]  # (77862, 3)
# print('Input shape',n_inputs)
# model = make_model(n_inputs, metrics=METRICS)
# print(model.summary())
# EPOCHS=300 #200
# BATCH_SIZE=512# 2048
# model_history = model.fit(X_train, y_train,
#                           validation_data = (X_test, y_test),
#                           epochs=EPOCHS,
#                           batch_size=BATCH_SIZE,
#                           verbose=1,
#                           callbacks=[early_stopping],
#                           shuffle=True)
#
# # plot training history
# plt.plot(model_history.history['loss'], label='train')
# plt.plot(model_history.history['val_loss'], label='test')
# plt.legend()
# plt.grid()
# plt.show()
#
# # plot training history
# plt.plot(model_history.history['prc'], label='train')
# plt.plot(model_history.history['val_prc'], label='test')
# plt.legend()
# plt.grid()
# plt.show()
#
# y_test_pred = model.predict(X_test)
# print('AUPRC',average_precision_score(y_test, y_test_pred))
# print('AUROC',roc_auc_score(y_test, y_test_pred))
#
# print()
# print(confusion_matrix(y_test, y_test_pred > 0.5))
# print(classification_report(y_test, y_test_pred > 0.5))
# plot_precision_recall_vs_threshold(y_test, y_test_pred)