# import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Load Data
company = '2330.TW'
start = dt.datetime(2012, 1, 1)
end = dt.datetime(2023, 11, 30)
stock_price_data = yf.download(company, start=start, end=end)

# visualize the closing price history
plt.figure(figsize=(20, 10))
plt.title('TSMC Close Price History')
plt.plot(stock_price_data['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price TWD (NTD$)', fontsize=18)
plt.savefig('TSMC_Close_Price_History.png')
# plt.show()

# create a new dataframe with only the 'Close' column
data = stock_price_data.filter(['Close'])

# convert the dataframe to a numpy array
dataset = data.values

# get the number of rows to train the model on
training_data_len = math.ceil(len(dataset) * 0.8)

# create the training and testing dataset
training_data = dataset[:training_data_len, :]
testing_data = dataset[training_data_len-60:, :]

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(training_data)
scaled_training_data = scaler.transform(training_data)
scaled_testing_data = scaler.transform(testing_data)

# split the data into x_train and y_train datasets
x_train = []
y_train = []
for i in range(60, len(scaled_training_data)):
    x_train.append(scaled_training_data[i - 60:i, 0])
    y_train.append(scaled_training_data[i, 0])

# convert the x_train and y_train to numpy arrays
x_train = np.array(x_train)
y_train = np.array(y_train)

# reshape the data (LSTM model expects 3D data (number of samples, number of time steps, number of features))
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # add LSTM layer with 50 neurons, return sequences = True because we will add another LSTM layer
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))  # add Dense layer with 25 neurons
model.add(Dense(1))  # add Dense layer with 1 neuron

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# create the testing dataset
# create a new array containing scaled values from index 1543 to 2002
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(scaled_testing_data)):
    x_test.append(scaled_testing_data[i - 60:i, 0])

# convert the data to a numpy array
x_test = np.array(x_test)

# reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get the models predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)  # undo scaling

# get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(rmse)

# plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# visualize the data
plt.figure(figsize=(20, 10))
plt.title('TSMC Close Price History')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price TWD (NTD$)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.savefig('TSMC_Price_Prediction.png')
# plt.show()

# show the valid and predicted prices
print(valid)
valid.to_csv('TSMC_Price_Prediction.csv')
