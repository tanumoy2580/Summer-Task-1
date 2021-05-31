# Training and Saving Model

from sklearn.linear_model import LinearRegression
md=LinearRegression()
import numpy as ny
a1=ny.array([1,2,3,4,5])
a2=ny.array([21,42,63,84,105])
import pandas as p
data=p.read_csv('/root/Salary_Data.csv')
a1=data.YearsExperience
a2=data.Salary
a1=ny.array(a1)
a1=a1.reshape(-1,1)
a2=ny.array(a2)
a2=a2.reshape(-1,1)
md.fit(a1,a2)
md.predict([[2]])
import joblib
joblib.dump(md,'/root/td.pkl')


#Loading Model for Prediction

td=joblib.load('/root/td.pkl')
td.predict([[2]])