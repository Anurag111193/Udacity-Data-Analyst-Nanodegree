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

menDt = dt[dt.Sex == 0]

femDt = dt[dt.Sex == 1]

menprbclass= menDt.groupby('Pclass').Survived.mean() 
print "Male Survival Rate"
print menprbclass * 100

#femDt = dt[dt.Sex == 1]
femprbclass= femDt.groupby('Pclass').Survived.mean()
print "Female Survival Rate"
print femprbclass * 100

menprbclass= menDt.groupby('ChildPass').Survived.mean() 
print "Male Survival Rate based on age group"
print menprbclass * 100

#femDt = dt[dt.Sex == 1]
femprbclass= femDt.groupby('ChildPass').Survived.mean()
print "Female Survival Rate based on age group"
print femprbclass * 100


femprbclass= dt.groupby('Parch').Survived.mean()
print "Survival Rate based on siblings"
print femprbclass * 100
















