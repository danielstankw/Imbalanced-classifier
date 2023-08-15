import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
import seaborn as sns
# TODO combine 8 df into one big for more data
dff1 = pd.read_csv('old_error_1cm/circle_0.001.csv')
dff2 = pd.read_csv('old_error_1cm/circle_0.0008.csv')
dff3 = pd.read_csv('old_error_1cm/circle_0.0006.csv')

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

# We check distribution of the features to see how skewed they are
fig, ax = plt.subplots(1, 3, figsize=(18, 4))

Fz = df['Fz'].values
Mx = df['Mx'].values
My = df['My'].values

sns.distplot(Fz, ax=ax[0], color='r')
ax[0].set_title('Distribution of Fz', fontsize=14)
ax[0].set_xlim([min(Fz), max(Fz)])

sns.distplot(Mx, ax=ax[1], color='b')
ax[1].set_title('Distribution of Mx', fontsize=14)
ax[1].set_xlim([min(Mx), max(Mx)])

sns.distplot(My, ax=ax[2], color='b')
ax[2].set_title('Distribution of My', fontsize=14)
ax[2].set_xlim([min(My), max(My)])
# plt.show()

# ['time', 'Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Mx', 'My', 'Case', 't_contact']
feature_names = ['Fz', 'Mx', 'My']

X = df[feature_names]
y = df.Case

print(f"Ratio of 1's in training set {y.sum()/len(y)*100}% and count is: {y.sum()}")
print(f"Ratio of 0's in training set {100 - y.sum()/len(y)*100}% and count is: {len(y)-y.sum()}")

#TODO: SPLITTING THE DATA

#
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# sns.histplot(ax=axes[0], data=impedance_one, x="Fz", kde=True, color='blue')
# axes[0].set_title('Case1: possible insertion using impedance')
# sns.histplot(ax=axes[1], data=overlap_one, x="Fz", kde=True, color='red')
# axes[1].set_title('Case2: Overlap but not possible insertion')
# sns.histplot(ax=axes[2], data=no_overlap_one, x="Fz", kde=True, color='C2')
# axes[2].set_title('Case3: No Overlap')
# plt.show()


# Stratified sampling aims at splitting a data set so that each split is similar with respect to something.
sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

X_train, X_test, y_train, y_test = None, None, None, None

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_X_train, original_X_test = X.iloc[train_index], X.iloc[test_index]
    original_y_train, original_y_test = y.iloc[train_index], y.iloc[test_index]

size_train = len(original_X_train) + len(original_y_train)
size_test = len(original_X_test) + len(original_y_test)
print(f"Ratio of 1 in train {y_train.sum()/len(y_train)*100} ")
print(f"Ratio of 1 in test {original_y_test.sum()/len(original_y_test)*100}")

'''We have same distribution of labels in train and test data!'''

X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train, y_train)
# A higher k (number of folds) means that each model is trained on a larger training set
# and tested on a smaller test fold. In theory, this should lead to a
# lower prediction error as the models see more of the available data.
cv = 10
print('Accuracy after training', cross_val_score(rfc, X_train, y_train, cv=cv, scoring='accuracy'))
print('Dumb classifier accuracy', 1 - y_test.sum()/len(y_test))
# todo: we are dealing with unbalanced dataset/ skewed dataset

# TODO: WE USE EVERYTHING ON THE TRAIN SET CAUSE TEST WILL BE USED ONLY AT THE END TO VALIDATE RESULTS!
# y_train_pred = rfc.predict(X_train)
y_train_pred = cross_val_predict(rfc, X_train, y_train, cv=cv)

# Calculate the confusion matrix
confusion = confusion_matrix(y_train, y_train_pred)
print(f"Training Confusion matrix:\n{confusion}")
# cv=10
# [[159663    186]
#  [  2650  55413]]
# cv=5
# [[133066  26783]
#  [ 22122  35941]]
# perfect
# array([[159849,      0],
#        [     0,  58063]])
# simple predict
# [[159849      0]
#  [     2  58061]]

print('precision score', precision_score(y_train, y_train_pred))
print('recall score', recall_score(y_train, y_train_pred))
print('F1 score', f1_score(y_train, y_train_pred))


# 1 is an overlap class
# 0 is no overlap class

# FP: we identified no overlap as an overlap
# FN: we identified overlap as no overlap

# we wish to identify the overlap as no overlap as it will still yield insertion -> higher precision
# we want to increase the threshold increases precision


