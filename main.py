import torch
import zipfile
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pytorch_lightning as pl
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, confusion_matrix

# with zipfile.ZipFile('career-con-2019.zip', 'r') as zip_ref:
#     zip_ref.extractall("")

pl.seed_everything(42)

X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')

# series_id defined number of the series as each series is composed of multiple smaller series
# labels are one per series

# plt.figure()
# y_train.surface.value_counts().plot(kind='bar')
# plt.xticks(rotation=45)
# plt.show()

# TODO: we obtained imbalanced labels, ensamble etc to balance dataset

## Preprocess
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train.surface)

print(encoded_labels[:5])
print(label_encoder.classes_)

y_train['label'] = encoded_labels

FEATURE_COLUMNS = X_train.columns.tolist()[3:]
print(FEATURE_COLUMNS)

# segmentation was done for us, need to do it!
# each series is split into 128 datapoints


print(1)