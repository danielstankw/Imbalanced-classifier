import numpy as np
from tensorflow import keras
import pandas as pd
import time
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


# ['time', 'Fx', 'Fy', 'Fz', 'Vx', 'Vy', 'Mx', 'My', 'Case', 't_contact']
feature_names = ['Fz', 'Mx', 'My']

X = df[feature_names].to_numpy()
y = df.Case.to_numpy()


model = keras.models.load_model("no_window")
# self.model = pickle.load(open('RFClassifier.sav', 'rb'))
threshold_precision = 0.788
temp = np.array([0.2, 0.72, 1.3]).reshape(1, 3)

time_vec = []
for i in range(10):
    t_init = time.time()

    y_score = model(temp).numpy()
    # y_scores = model.predict(temp)

    dt = time.time()-t_init
    time_vec.append(dt)

print(sum(time_vec)/len(time_vec))