import csv
import os

import seaborn as sns
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

read_all = False
if read_all:
    """Code responsible for merging datasets"""
    df = pd.DataFrame()
    # loop through file names
    # for i in range(1, 101):
    for i in range(1, 101):
        file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection/ep{}.csv".format(i)
        df_new = pd.read_csv(file_name)
        df = pd.concat([df, df_new], ignore_index=True)

    filename = "all_runs.csv"
    filepath = os.path.join('/home/user/Desktop/Simulation_n5/robosuite/data_collection', filename)
    df.to_csv(filepath)
    print('Done')

collect_positive = False
if collect_positive:
    """Creates new df only from the values that contains labels with overlap"""
    df = pd.DataFrame()
    for i in range(1, 101):
        file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection/ep{}.csv".format(i)
        df_new = pd.read_csv(file_name)
        if df_new.case.sum() > 0:
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            print(i)
            continue
    filename = "all_with_label.csv"
    filepath = os.path.join('/home/user/Desktop/Simulation_n5/robosuite/data_collection', filename)
    df.to_csv(filepath)
    print('Done')

#
file_name = "/home/user/Desktop/Simulation_n5/robosuite/data_collection/all_with_label.csv"
df_new = pd.read_csv(file_name)
print()
total = len(df_new['case'])
pos = np.sum(df_new['case'])
neg = total - pos
print(f"Ratio of 1's in full data {(pos/total)*100}% and count is: {pos}")
print(f"Ratio of 0's in training set {(neg/total) * 100}% and count is: {neg}")

#
# sample = pd.read_csv("/home/user/Desktop/Simulation_n5/robosuite/data_collection/ep2.csv")
#
# plt.figure()
# plt.plot(sample.t, sample.fz)
#
# plt.figure()
# plt.scatter(sample.t, sample.case)
# plt.show()