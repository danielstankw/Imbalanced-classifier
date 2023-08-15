import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# TODO combine 8 df into one big for more data
dff = pd.read_csv('spiral_measurement1.csv')
# 3-no overlap, 2-overlap but not insert, 1-enough to insert
# we will combine 3 and 2 into 0 label
dff['Case'] = dff['Case'].replace([3, 2], 0)
# use only dataframe after contact
t_cont = dff['t_contact'][0]
df = dff[dff.time > t_cont]

# features = df[df.time > t]s used for training
print(df.columns)
feature_names = ['Fz', 'Mx', 'My']

# TODO try random forest
# TODO: due to imbalanced dataset we wont use accuracy as metrics
# try copies of instances from under represented class/ delete instanced from over represented
# try random forest
# stratified sampling
# area under ROC or precission and recall

X = df[feature_names]
y = df.Case

print('No Overlap case', round(df['Case'].value_counts()[0]/len(df) * 100, 2), '% of the dataset')
print('Overlap case', round(df['Case'].value_counts()[1]/len(df) * 100, 2), '% of the dataset')
# 98% vs 1.51% very unbalanced - > used SMOTE

# split to train and test > sampling on the training alone
# https://www.kaggle.com/general/238949
# https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook

'''Splitting the data:
We split before re-sampling. We want to test model on original non-re-sampled data
Fit model with sampled data/ test on original'''

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# Check the Distribution of the labels

# Turn into an array
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# See if both the train and test label distribution are similarly distributed
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))


'''Under-sampling:
Removal of data to get better dataset

1. Determine how imbalanced dataset is
2. Bring the labels to same amount
3. Shuffle

Issue: lots of information loss as we cut out the existing data'''
print('No Overlap case', round(df['Case'].value_counts()[0]/len(df) * 100, 2), '% of the dataset')
print('Overlap case', round(df['Case'].value_counts()[1]/len(df) * 100, 2), '% of the dataset')
















#
#
#
# # smote: over-sampling approach -> minority is over-sampled by creating synthetic examples
# sample = 'under'
#
# # TODO: try without re-balancing!
# if sample == 'over':
#     # We can see that what SMOTE does is create new synthetic data points which
#     # are situated randomly along line segments that connect existing minority class points
#     X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
# else:
#     # randomly select subset of data for target classes
#     X_resampled, y_resampled = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)
#
#
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# tf.random.set_seed(42)
#
# n_features = X_train_scaled.shape[1]
# print()
#
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(n_features,)))
# model.add(tf.keras.layers.Dense(24, activation='relu'))
# model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#
# model.summary()
# # compile the model
# model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.03),
#               loss=tf.keras.losses.binary_crossentropy,
#               metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
#                        tf.keras.metrics.Precision(name='precision'),
#                        tf.keras.metrics.Recall(name='recall')])
#
#
# #
# history = model.fit(X_train_scaled, y_train, epochs=50, validation_split=0.1)
#
# pd.DataFrame(history.history).plot(figsize=(8, 5))
# plt.grid(True)
# plt.gca().set_ylim(0, 1)
# plt.show()
