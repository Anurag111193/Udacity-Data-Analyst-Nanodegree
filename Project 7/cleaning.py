import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
dt = pd.read_csv('C:/wamp/www/Project 7/data.csv')
dt['Sex'].replace(['male','female'],[0,1],inplace=True)
def child_pass(x):
    if x < 16:
        return 'Child'
    if x > 60:
        return 'Elder'
    else :
        return 'Adult'

dt["ChildPass"] = pd.Series(dt["Age"].apply(child_pass), index=dt.index)
dt.to_csv('C:/wamp/www/Project 7/data1.csv')
