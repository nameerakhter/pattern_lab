import pandas as pd
import numpy as np
import sklearn.linear_model as lm

#read data
df = pd.read_csv('D:/pattern recognition and anomaly detection/lab-2/NIFTY22JANFUT.csv')
df.set_index('dateTime', inplace=True)

#add new column with previous day close
df['PrevClose'] = df['close'] - df['close'].shift(1)

#add new column with previous day volume
df['PrevVolume'] = df['volume'] -df['volume'].shift(1)

#df['PrevVOpen'] = df['open'] -df['open'].shift(1)

#df['Prevhigh'] = df['high'] -df['high'].shift(1)

#df['Prevlow'] = df['low'] -df['low'].shift(1)

#drop rows with NaN
df.dropna(inplace=True)

#train model
model = lm.LinearRegression()
model.fit(df[['open','low', 'high', 'PrevClose', 'PrevVolume']], df['close'])
#predict
df['Predicted'] = model.predict(df[['open','low', 'high', 'PrevClose', 'PrevVolume']])
#calculate error
df['Error'] = df['Predicted'] - df['close']
#calculate mean absolute error
mae = np.mean(np.abs(df['Error']))
print('Mean Absolute Error: ', mae)
#calculate mean squared error
mse = np.mean(np.square(df['Error']))
print('Mean Squared Error: ', mse)
#print coefficients
print('Coefficients: ', model.coef_)