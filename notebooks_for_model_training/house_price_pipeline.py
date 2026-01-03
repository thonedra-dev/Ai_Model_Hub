import pandas as pd
df = pd.read_csv('datasets/melb_data.csv')
df.head(100)

Raw_X = df[['Rooms', 'Distance', 'Bathroom', 'Landsize', 'BuildingArea',
       'Lattitude', 'Longtitude', 'Car']]

y = df.Price

from sklearn.impute import SimpleImputer
import numpy as np
X = Raw_X.copy()


Q1 = df['BuildingArea'].quantile(0.25)
Q3 = df['BuildingArea'].quantile(0.75)
IQR = Q3 - Q1
upper = Q3 + 1.5 * IQR
X['BuildingArea'] = np.where(df['BuildingArea'] > upper, upper, df['BuildingArea'])
X['BuildingArea'] = SimpleImputer(strategy='median').fit_transform(X[['BuildingArea']])

upper = df['Landsize'].quantile(0.99)   # cap at 99th percentile
X['Landsize'] = np.where(X['Landsize'] > upper, upper, X['Landsize'])

X['Car'] = SimpleImputer(strategy='median').fit_transform(X[['Car']])

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

rf_model = RandomForestRegressor(n_estimators=200, max_depth=20, max_features='sqrt', random_state=1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2_score = r2_score(y_test, rf_pred)
print(f"n_estimators: 590, Depth: 20 -> MAE Test: {rf_mae:.2f}, R2 Test: {rf_r2_score:.3f}")



from joblib import dump
dump(rf_model, "house_price__pipeline.joblib")