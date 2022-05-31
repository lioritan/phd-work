import pickle
import pandas as pd
import numpy as np


with open(r"D:\GitHub\phd-work\curriculum_code\results\2021-05-11 01 mixed\difficulty\MBPendulum-v2\data_40.pkl", "rb") as fptr:
    estimates, params = pickle.load(fptr)

tasks = pd.DataFrame(params)

print(np.corrcoef(estimates.mean(axis=1), tasks[0]))#, np.corrcoef(estimates.mean(axis=1), tasks[1]), np.corrcoef(estimates.mean(axis=1), tasks[2]), np.corrcoef(estimates.mean(axis=1), tasks[3]))
print("_________________________________")
print(estimates.mean(), estimates.std())
print("_________________________________")
print(estimates.mean(axis=1)[estimates.mean(axis=1)>0].shape, estimates.mean(axis=1)[estimates.mean(axis=1)>100].shape)

print(np.corrcoef(estimates.mean(axis=1), tasks))
estimates.mean(axis=1)[estimates.mean(axis=1)>0].shape # VS 625
estimates.mean(axis=1)[estimates.mean(axis=1)>100].shape
print(666)
