#%%
import pandas as pd
import torch
import torch.nn as nn
import seaborn as sns

#%% importando dataset
data_file = 'car_data.csv'
data = pd.read_csv(data_file)
data.head()
# %%
