# -*- coding: utf-8 -*-
"""mushroom(final)

Automatically generated by Colaboratory.

"""

import io 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from google.colab import files         
uploaded = files.upload()

data= pd.read_csv(io.BytesIO(uploaded['mushrooms.csv']))
data

data.isnull().sum()

columns = data.columns.to_list()
print("*column by column data distributions*\n")
for col in columns:
    print(col,"\n",data[col].value_counts(),"\n\n")

total = float(len(data[columns[0]]))
plt.figure(figsize=(4,4))
sns.set(style="dark")
i = sns.countplot(data[columns[0]])
for j in i.patches:
    height = j.get_height()
    i.text(j.get_x()+j.get_width()/2.,height + 3,'{:1.2f}'.format(height/total),ha="center")
plt.title("Target ", fontsize = 15)
plt.show()

for col in columns[1:]:
    plt.figure(figsize=(7,4))
    sns.countplot(x=col , data=data ,palette='cubehelix')
    plt.title(col, fontsize=15)
    plt.show()
    print("% of total:")
    print(round((data[col].value_counts()/data.shape[0]),4)*100)

for col in columns[1:]: 
    plt.figure(figsize=(15,10))
    plt.subplot(2,3,1)
    sns.countplot(x=col ,hue='class', data=data ,palette='cubehelix')
    plt.xlabel(col, fontsize=15)
    plt.legend(loc='upper right')

[ pd.pivot_table(data, index=[col,"class"], aggfunc = {col:np.count_nonzero}) for col in columns[1:]]
