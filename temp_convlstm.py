## Import Libraries
import os
import numpy as np
import xarray as xr
import tensorflow as tf
tf.random.set_seed(42)
from supervised import supervised_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ConvLSTM2D,BatchNormalization,Conv3D

# Access the Required GPU
os.environ['CUDA_VISIBLE_DEVICES']="7"

# Access the required file paths and save them as variable
with open('convlstm_path.txt') as f:
    for i,line in enumerate(f):
        if i == 1:
            temp_path = line.strip()
        if i == 3:
            check_point_path = line.strip()
        if i == 5:
            log_dir_path = line.strip()
        if i == 7:
            model_save_path = line.strip()

## Read Temperature File
ds = xr.open_dataarray(temp_path)

## Data Scaling
max_temp = ds.max()    # Maximum Temperature
min_temp = ds.min()    # Mininmum Temperature

ds_scaled = (ds - min_temp) /(max_temp - min_temp)   # Normalization of Xarray Dataset

## Train Validation Test Split ##
X_train, y_train  =  supervised_split(ds_scaled.sel(time = slice('1959-01-01','2000-12-31')),5,5)
X_valid, y_valid  =  supervised_split(ds_scaled.sel(time = slice('2001-01-01','2010-12-31')),5,5)
# X_test , y_test   =  supervised_split(ds_scaled.sel(time = slice('2011-01-01','2021-12-31')),5,5)

# Check the Shape #
print('The shape of (X_train,y_train) is <=> ', X_train.shape,y_train.shape)
print('The shape of (X_valid,y_valid) is <=> ', X_valid.shape,y_valid.shape)
# print('The shape of (X_test,y_test)   is <=> ', X_test.shape,y_test.shape)


## Tensorflow Model ##
model = Sequential()

model.add(ConvLSTM2D(
    filters=16,kernel_size=(3,3),padding = 'same',
    data_format='channels_last',activation='relu',
    return_sequences=True,input_shape = (5,129,135,1)))

model.add(BatchNormalization())
    
model.add(ConvLSTM2D(
    filters=16,kernel_size=(3,3),padding = 'same',
    data_format='channels_last',activation='relu',
    return_sequences=True))

model.add(BatchNormalization())

model.add(Conv3D(
    filters=15,kernel_size=(3,3,3),
    activation='relu',padding='same',data_format='channels_last'))


model.add(Conv3D(
    filters=1,kernel_size=(3,3,3),
    activation='relu',padding='same',data_format='channels_last'))

# Adam Optimizer
Adam = tf.keras.optimizers.Adam(learning_rate = 10**-4)

# Compile the Model Above
model.compile(loss = 'mean_squared_error',metrics = ['mse','mae'],optimizer=Adam)

# Save the Checkpoint for best model.
check_point = tf.keras.callbacks.ModelCheckpoint(
    check_point_path,
    # save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
)

# Tensorboard for Visualization of Loss vs epochs
tensorboard = tf.keras.callbacks.TensorBoard(
    log_dir= log_dir_path,
)

# Fit the model on train and validation data #
model.fit(X_train,y_train,
epochs=100,validation_data=(X_valid,y_valid),verbose = 1,
callbacks=[tensorboard,check_point],batch_size=64)

# Save the Model with weights #
model.save(model_save_path)

print('-'*100)
print('Model is Saved !!!')
print('-'*100)