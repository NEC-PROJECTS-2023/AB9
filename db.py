import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

crop_data = pd.read_csv("crop_production.csv")

crop_data.isnull().sum()
crop_data = crop_data.dropna()

lb = preprocessing.LabelEncoder()
crop_data['State_Name'] = lb.fit_transform(crop_data['State_Name'])
crop_data['Season'] = lb.fit_transform(crop_data['Season'])
crop_data['Crop'] = lb.fit_transform(crop_data['Crop'])

from sklearn.model_selection import train_test_split

x = crop_data.drop(["Production", 'District_Name'], axis=1)
y = crop_data["Production"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)
model = RandomForestRegressor()
model.fit(x_train, y_train)

pickle.dump(model, open('model.pkl', 'wb'))
