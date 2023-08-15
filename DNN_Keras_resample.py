import pandas as pd
import numpy as np
import sklearn
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
import plotly.express as px
SAVE = False
LOAD = False

keras.backend.clear_session()


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


def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, threshold = roc_curve(labels, predictions)

    plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5, 100])
    plt.ylim([0.0, 100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
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

neg, pos = np.bincount(df['Case'])
total = neg + pos
print('neg', neg)
print('pos', pos)

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

tf.random.set_seed(42)

n_inputs = X_train_res.shape[1]  # (77862, 3)

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
    # nn.add(keras.layers.Dropout(0.5))
    # nn.add(keras.layers.Dense(32, activation='relu'))
    # nn.add(keras.layers.Dense(64, activation='relu'))
    nn.add(keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias))

    nn.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=metrics)
    return nn

path_name = 'Weights_folder_3/Weights'
# model = make_model()
# model.save_weights(path_name)

EPOCHS = 50  # 100
BATCH_SIZE = 208#  1024  # 2048

use_resample = True

# TRAINING RESAMPLED MODEL
if use_resample:
    model_res = make_model()
    model_res.load_weights(path_name)
    model_res_history = model_res.fit(
        X_train_res,
        y_train_res,
        epochs=EPOCHS,
        validation_split=0.2,
        shuffle=True)

    plot_loss(model_res_history, "SMOTE Trained Model Loss", col='red')
    plt.show()
    plot_metrics(model_res_history)
    plt.show()

    train_pred_res = model_res.predict(X_train_res, batch_size=200)
    test_pred_res = model_res.predict(original_X_test, batch_size=200)

    print('Classification Report for: Resampled TRAIN SET')
    print(classification_report(y_train_res, train_pred_res > 0.5))
    print('Classification Report for: Resampled TEST SET')
    print(classification_report(original_y_test, test_pred_res > 0.5))

breakpoint()
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

path_name = 'Weights_folder_3/Weights'

model_orig = make_model()
model_orig_history = model_orig.fit(
    original_X_train,
    original_y_train,
    epochs=EPOCHS,
    validation_split=0.2,
    shuffle=True,
    class_weight=class_weight)

plot_loss(model_orig_history, "Original Trained Model Loss", col='red')
plt.show()
plot_metrics(model_orig_history)
plt.show()

train_pred_ori = model_orig.predict(original_X_train, batch_size=200)
test_pred_ori = model_orig.predict(original_X_test, batch_size=200)

print('Classification Report for: Original TRAIN SET')
print(classification_report(original_y_train, train_pred_ori > 0.5))
print('Classification Report for: Original TEST SET')
print(classification_report(original_y_test, test_pred_ori > 0.5))
y_scores = model_orig.predict(original_X_test)
precisions, recalls, thresholds = precision_recall_curve(original_y_test, y_scores)
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
#

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
#
# plot_roc("Train Baseline", original_y_train, train_pred_ori, color='blue')
# plot_roc("Test Baseline", y_test, test_pred_ori, color='blue', linestyle='--')
# plt.legend(loc='lower right')
# plt.show()
# TODO: for tuning
# plot_roc_interactive('Train', original_y_train, train_pred_ori)
# plot_roc_interactive('Test', y_test, test_pred_ori)


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
