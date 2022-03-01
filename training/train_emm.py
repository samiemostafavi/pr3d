import warnings
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
import os.path
import math
import bisect
import tensorflow as tf
import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork

""" Load training data stored in .npz format """

# Path
path = 'saves/'

""" import the train datasets """

batch_size = 10000

file_addrs = [ 'traindata_p1_10k.npz' \
                ,'traindata_p2_10k.npz' \
                ,'traindata_p3_10k.npz' ]

npzfile = np.load(path + file_addrs[0])
train_data = npzfile['arr_0']
meta_info = npzfile['arr_1']
batch_size = int(meta_info[0])
p1_ndim_x = int(meta_info[1])
predictor_num = int(meta_info[2])
p1_n_replicas = int(meta_info[3])
print('Predictor-%d training data loaded from .npz file. Rows: %d ' %(predictor_num,len(train_data[:,0,0])) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % p1_ndim_x)
train_data_p1 = train_data

npzfile = np.load(path + file_addrs[1])
train_data = npzfile['arr_0']
meta_info = npzfile['arr_1']
batch_size = int(meta_info[0])
p2_ndim_x = int(meta_info[1])
predictor_num = int(meta_info[2])
p2_n_replicas = int(meta_info[3])
print('Predictor-%d training data loaded from .npz file. Rows: %d ' %(predictor_num,len(train_data[:,0,0])) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % p2_ndim_x)
train_data_p2 = train_data

npzfile = np.load(path + file_addrs[2])
train_data = npzfile['arr_0']
meta_info = npzfile['arr_1']
batch_size = int(meta_info[0])
p3_ndim_x = int(meta_info[1])
predictor_num = int(meta_info[2])
p3_n_replicas = int(meta_info[3])
print('Predictor-%d training data loaded from .npz file. Rows: %d ' %(predictor_num,len(train_data[:,0,0])) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % p3_ndim_x)
train_data_p3 = train_data

""" create the directory """

# Path
path = 'saves/trained_models/'
# Get the directory name from the specified path
dirname = os.path.dirname(path)
# create the dir
os.makedirs(dirname, exist_ok=True)


""" create the EMM predictor-1 models and train them """
"""
predictor_num = 1
n_epoch_emm = 12000
p1_n_replicas = 1
for n in range(p1_n_replicas):
    MODEL_NAME = 'trained_'+'emm_p'+ str(predictor_num) + '_s' + str(int(batch_size/1000)) + '_r' + str(n)
    train_data = train_data_p1[:,:,n]
    Y = train_data[:,0]
    X = train_data[:,1:]

    model = NoNaNGPDExtremeValueMixtureDensityNetwork(MODEL_NAME, ndim_x=p1_ndim_x, n_centers=2, ndim_y=1, n_training_epochs=n_epoch_emm, hidden_sizes=(16, 16),verbose_step=math.floor(n_epoch_emm/10), weight_decay=0, learning_rate=5e-4,epsilon=1e-6,l2_reg=0, l1_reg=0,dropout=0.0)
    model.fit(X, Y)

    with open(path + MODEL_NAME + '.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
"""


""" create the EMM predictor-2 models and train them """
"""
predictor_num = 2
n_epoch_emm = 12000
p2_n_replicas = 1
for n in range(p2_n_replicas):
    MODEL_NAME = 'trained_'+'emm_p'+ str(predictor_num) + '_s' + str(int(batch_size/1000)) + '_r' + str(n)
    train_data = train_data_p2[:,:,n]
    Y = train_data[:,0]
    X = train_data[:,1:]

    model = NoNaNGPDExtremeValueMixtureDensityNetwork(MODEL_NAME, ndim_x=p2_ndim_x, n_centers=2, ndim_y=1, n_training_epochs=n_epoch_emm, hidden_sizes=(16, 16),verbose_step=math.floor(n_epoch_emm/10), weight_decay=0, learning_rate=5e-4,epsilon=1e-6,l2_reg=0, l1_reg=0,dropout=0.0)
    model.fit(X, Y)

    with open(path + MODEL_NAME + '.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
"""

""" create the EMM predictor-3 models and train them """
predictor_num = 3
n_epoch_emm = 12000 # 7000
p3_n_replicas = 1
for n in range(p3_n_replicas):
    MODEL_NAME = 'trained_'+'emm_p'+ str(predictor_num) + '_s' + str(int(batch_size/1000)) + '_r' + str(n)
    train_data = train_data_p3[:,:,n]
    Y = train_data[:,0]
    X = train_data[:,1:]

    model = NoNaNGPDExtremeValueMixtureDensityNetwork(MODEL_NAME, ndim_x=p3_ndim_x, n_centers=2, ndim_y=1, n_training_epochs=n_epoch_emm, hidden_sizes=(16, 16),verbose_step=math.floor(n_epoch_emm/10), weight_decay=0, learning_rate=5e-4,epsilon=1e-6,l2_reg=0, l1_reg=0,dropout=0.0)
    model.fit(X, Y)

    with open(path + MODEL_NAME + '.pkl', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)