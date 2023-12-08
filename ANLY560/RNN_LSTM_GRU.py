import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import RMSprop
global lag            #Forecast time lag
lag=7
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics
from keras import losses
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('BKNG.csv')
data.head()

nrow=data.shape[0]
train_index=list(range(int(0.7*(nrow-lag))))
validation_index=list(range(int(0.7*(nrow-lag)),int(0.9*(nrow-lag))))
test_index=list(range(int(0.9*(nrow-lag)),(nrow-lag)))

def generate_X_y(data,ex_rate):
    tmp=data[ex_rate]
    print('Raw data mean:',np.mean(tmp),'\nRaw data std:',np.std(tmp))
    tmp=(tmp-np.mean(tmp))/np.std(tmp)

    X=np.zeros((nrow-lag,lag))
    for i in range(nrow-lag):X[i,:lag]=tmp.iloc[i:i+lag]
    y=np.array(tmp[lag:]).reshape((-1,1))
    return (X,y)

X,y=generate_X_y(data,'Adj Close')
X_train,y_train=X[train_index,:],y[train_index,:]
X_validation,y_validation=X[validation_index,:],y[validation_index,:]
X_test,y_test=X[test_index,:],y[test_index,:]#Raw data mean: 2301.833445070234; Raw data std: 140.49789024716577

def get_Mae_benchmark(X,y):
    mean_benchmark=np.mean(np.abs(np.mean(X,0)-y))
    drift_benchmark=np.mean(np.abs(X[:,-1]-y))
    return(mean_benchmark,drift_benchmark)

mean_validate_benchmark,drift_validate_benchmark=get_Mae_benchmark(X_validation,y_validation)
mean_test_benchmark,drift_test_benchmark=get_Mae_benchmark(X_test,y_test)
print('Mean Validate MAE Benchmark:',mean_validate_benchmark,'; ','Drift Validate MAE Benchmark:',drift_validate_benchmark)
print('Mean Test MAE Benchmark:',mean_test_benchmark,'; ','Drift Test MAE Benchmark:',drift_test_benchmark)


def training_performance(model, training_history, epochs):
    test_MAE = np.mean(np.abs(y_test - model.predict(X_test.reshape(-1, lag, 1))))

    timestep = range(1, epochs + 1)

    plt.figure(figsize=(10, 8), facecolor='ghostwhite')
    plt.subplot(2, 1, 1)
    plt.plot(timestep, np.log(training_history.history['val_mae']), 'b', label='Validation MAE')
    plt.plot(timestep, np.log(training_history.history['mae']), 'bo', label='Training MAE')
    plt.hlines(np.log(drift_validate_benchmark), xmin=timestep[0], xmax=timestep[-1], colors='coral',
               label='Validation Drift Benchmark')
    plt.hlines(np.log(mean_validate_benchmark), xmin=timestep[0], xmax=timestep[-1], colors='lightblue',
               label='Validation Mean Benchmark')
    plt.hlines(np.log(test_MAE), xmin=timestep[0], xmax=timestep[-1], colors='purple', label='Testing MAE')
    plt.ylabel('logged MAE')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.hlines(test_MAE, xmin=timestep[0], xmax=timestep[-1], colors='purple', label='Testing MAE')
    plt.hlines(drift_test_benchmark, xmin=timestep[0], xmax=timestep[-1], colors='coral', label='Test Drift Benchmark')
    plt.hlines(mean_test_benchmark, xmin=timestep[0], xmax=timestep[-1], colors='lightblue',
               label='Test Mean Benchmark')
    plt.ylabel('MAE')
    plt.legend(loc='right')
    plt.show()


def plot_prediction(model, ex_rate):
    tmp = data[ex_rate]

    plt.figure(figsize=(10, 8), facecolor='ghostwhite')
    prediction = model.predict(X_test.reshape(-1, lag, 1))
    prediction = prediction * np.std(tmp) + np.mean(tmp)
    y_true = y_test * np.std(tmp) + np.mean(tmp)
    plt.plot(list(range(len(prediction))), prediction, color='coral', label='Prediction')
    plt.plot(list(range(len(y_true))), y_true, color='purple', label='True Value')
    xticks = np.arange(0, len(y_true), 7)
    plt.xticks(xticks, labels=data.Date.iloc[test_index].iloc[xticks], rotation=90)
    plt.legend()
    plt.show()

#Simple RNN
optimizer=RMSprop()
model_RNN=models.Sequential()
model_RNN.add(layers.SimpleRNN(1,input_shape=(lag,1),activation='relu'))
model_RNN.add(layers.Dense(1))
model_RNN.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

history_RNN=model_RNN.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=16,epochs=80,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_RNN,history_RNN,80)
plot_prediction(model_RNN, 'Adj Close')

#Simple RNN-with Regularization
optimizer=RMSprop()
model_RNN_L1=models.Sequential()
model_RNN_L1.add(layers.SimpleRNN(1,input_shape=(lag,1),activation='relu',dropout=0.1,recurrent_dropout=0.05))
model_RNN_L1.add(layers.Dense(1))
model_RNN_L1.compile(optimizer=optimizer,loss='mse',metrics=['mae'])

history_RNN_L1=model_RNN_L1.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=16,epochs=300,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_RNN_L1,history_RNN_L1,300)
plot_prediction(model_RNN_L1,'Adj Close')

#LSTM
model_LSTM=models.Sequential()
model_LSTM.add(layers.LSTM(1,input_shape=(lag,1),activation='relu'))
model_LSTM.add(layers.Dense(1))
model_LSTM.compile(optimizer=RMSprop(),loss='mse',metrics=['mae'])

history_LSTM=model_LSTM.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=16,epochs=200,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_LSTM,history_LSTM,200)
plot_prediction(model_LSTM,'Adj Close')

#LSTM-with Regularization
model_LSTM_L1=models.Sequential()
model_LSTM_L1.add(layers.LSTM(1,input_shape=(lag,1),activation='relu',dropout=0.1,recurrent_dropout=0.05))
model_LSTM_L1.add(layers.Dense(1))
model_LSTM_L1.compile(optimizer=RMSprop(),loss='mse',metrics=['mae'])

history_LSTM_L1=model_LSTM_L1.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=16,epochs=600,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))
training_performance(model_LSTM_L1,history_LSTM_L1,600)
plot_prediction(model_LSTM_L1,'Adj Close')

#GRU
model_GRU=models.Sequential()
model_GRU.add(layers.LSTM(1,input_shape=(lag,1),activation='relu'))
model_GRU.add(layers.Dense(1))
model_GRU.compile(optimizer=RMSprop(),loss='mse',metrics=['mae'])

history_GRU=model_GRU.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=16,epochs=400,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_GRU,history_GRU,400)
plot_prediction(model_GRU,'Adj Close')

#GRU-with Regularization
model_GRU_L1=models.Sequential()
model_GRU_L1.add(layers.LSTM(1,input_shape=(lag,1),activation='relu',dropout=0.1,recurrent_dropout=0.05))
model_GRU_L1.add(layers.Dense(1))
model_GRU_L1.compile(optimizer=RMSprop(),loss='mse',metrics=['mae'])

history_GRU_L1=model_GRU_L1.fit(X_train.reshape(-1,lag,1),y_train.flatten(),batch_size=16,epochs=400,
                  validation_data=(X_validation.reshape(-1,lag,1),y_validation.flatten()))

training_performance(model_GRU_L1,history_GRU_L1,400)
plot_prediction(model_GRU_L1,'Adj Close')
