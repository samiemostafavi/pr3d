import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import math
import bisect
import tensorflow as tf
import warnings

from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork
from cde.data_collector import ParquetDataset



predictor_num = 3       #[1,2,3]
model_str = 'emm'       #['emm','gmm']
batch_size = 25000      # ?



# dataset parquet file:
file_addr = ['data/sim3hop_1_dataset_06_Sep_2021_11_20_40.parquet']
n_replicas = 1


""" import and create the train dataset into Numpy array """
FILE_NAME = 'traindata_p'+str(predictor_num)+'_'+str(int(batch_size/1000))+'k.npz'
training_dataset = ParquetDataset(file_addresses=file_addr,predictor_num=predictor_num)
train_data = training_dataset.get_data(batch_size,n_replicas)
ndim_x = len(train_data[0])-1
#print(np.shape(train_data))
meta_info = np.array([batch_size,ndim_x,predictor_num,n_replicas,file_addr])
np.savez(FILE_NAME, train_data, meta_info)
print('Predictor-%d dataset loaded from ' % predictor_num, file_addr,' and saved to '+FILE_NAME+'. Rows: %d ' % len(train_data[:,0,0]), ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)



""" Load training data stored in .npz format """
# modify the default parameters of np.load
npzfile = np.load(FILE_NAME,allow_pickle=True)
train_data = npzfile['arr_0']
meta_info = npzfile['arr_1']
batch_size = int(meta_info[0])
ndim_x = int(meta_info[1])
predictor_num = int(meta_info[2])
n_replicas = int(meta_info[3])
print('Predictor-%d training data loaded from .npz file. Rows: %d ' %(predictor_num,len(train_data[:,0,0])) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)


""" create the EMM predictor models and train them """
n_epoch_emm = 12000 # 7000
for n in range(n_replicas):
    MODEL_NAME = 'trained_'+'emm_p'+ str(predictor_num) + '_s' + str(int(batch_size/1000)) + '_r' + str(n)
    train_data = train_data[:,:,n]
    Y = train_data[:,0]
    X = train_data[:,1:]

    model = NoNaNGPDExtremeValueMixtureDensityNetwork(MODEL_NAME, ndim_x=ndim_x, n_centers=2, ndim_y=1, n_training_epochs=n_epoch_emm, hidden_sizes=(16, 16),verbose_step=math.floor(n_epoch_emm/10), weight_decay=0, learning_rate=5e-4,epsilon=1e-6,l2_reg=0, l1_reg=0,dropout=0.0)
    model.fit(X, Y)

    with open(MODEL_NAME + '.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)



""" load trained the model """
replica_num = 0
FILE_NAME = 'trained_'+model_str+'_p'+str(predictor_num)+'_s'+str(int(batch_size/1000))+'_r'+str(replica_num)+'.pkl'
if not os.path.isfile(FILE_NAME):
    print('No trained model found.')
    exit()
with open(FILE_NAME, 'rb') as input:
    if model_str is 'emm':
        model = NoNaNGPDExtremeValueMixtureDensityNetwork(name=model_str+str(predictor_num), ndim_x=4-predictor_num, ndim_y=1)
    else:
        model = MixtureDensityNetwork(name=model_str+str(predictor_num), ndim_x=4-predictor_num, ndim_y=1)
    model._setup_inference_and_initialize()
    model = pickle.load(input)


# State and budget 
xs = ([1,4,2], [4,2],    [2])
nus = (20,     18.34,    14.02)

print(model_str + '-p' + str(predictor_num) + ', batch_size=' + str(int(batch_size/1000)) + ', replica=' + str(replica_num) + ', x=' + str(xs[predictor_num-1]) + ', nu=' + str(nus[predictor_num-1]) + ', DVP: ' + str(model.tail(np.array([xs[predictor_num-1]]),np.array([nus[predictor_num-1]]))))

