import pandas as pd
import numpy as np

dff1 = pd.read_csv('circle_0.0008.csv')
dff1['Case'] = dff1['Case'].replace([3, 2], 0)
t_cont1 = dff1['t_contact'][0]
df= dff1[dff1.time > t_cont1]
feature_names = ['Fz', 'Mx', 'My']

X = df[feature_names].to_numpy()
y = df.Case.to_numpy()


n_features = 3
window_len = 3

row = X.shape[0] + 1 - window_len
col_len = n_features * window_len
new_x = np.zeros((row, col_len))
new_y = np.zeros((row, 1))

for i in range(len(new_x)):
    new_x[i] = X[i:i+3].reshape(1, col_len)[0][::-1]
    new_y[i] = y[i+2]

print()