import numpy as np

## Supervised Split
def supervised_split(ds,lag,lead):
    ds_new = ds.values.reshape(ds.shape[0], ds.shape[1], ds.shape[2],1)
    X = []
    y = []
    for i in range(ds.values.shape[0] - (lag + lead -1)):
        X.append(ds_new[i : i+lag])
        y.append(ds_new[i+lag : i+(lag+lead)])
    X = np.array(X,dtype='float32')
    y = np.array(y,dtype='float32')
    return X,y