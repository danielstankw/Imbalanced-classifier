import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

dff = pd.read_csv('spiral_measurement1.csv')

# use only dataframe after contact
t_cont = dff['t_contact'][0]
df = dff[dff.time > t_cont]

# features = df[df.time > t]s used for training
print(df.columns)
feature_names = list(df.columns)[1:6]
print(feature_names)

df_for_training = df[feature_names].astype(float)

# LSTM uses sigmoid and tanh which is sensitive to magnitude so values need to be normalized
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)


# We need to reshape input to n_samples x timesteps.
#
X_train = []
y_train = []

n_feature = 1 # number of timesteps we want to predict in the future
n_past = 14 # number of past timesteps used for prediction


print(1)










