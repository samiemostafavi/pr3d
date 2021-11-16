import numpy as np
import sys
import os
import os.path

#import tensorflow as tf
import pyarrow.parquet as pq

# if you run python inside the folder, then:
sys.path.insert(0, '../lib')
print(sys.path)

from cde.data_collector import ParquetDataset

""" Load or Create Project """
# this project uses the train dataset with arrival rate 0.9, and service rate 1

# Path
path = 'saves/'

# Get the directory name from the specified path
dirname = os.path.dirname(path)

# create the dir
os.makedirs(dirname, exist_ok=True)

""" training data """

# new file:
file_addr = ['../data/sim3hop_1_dataset_06_Sep_2021_11_20_40.parquet']
batch_size = 5000
n_replicas = 10


""" import and create the train dataset into Numpy array """
predictor_num = 1
FILE_NAME = 'traindata_p'+str(predictor_num)+'_'+str(int(batch_size/1000))+'k.npz'
training_dataset = ParquetDataset(file_addresses=file_addr,predictor_num=predictor_num)
train_data = training_dataset.get_data(batch_size,n_replicas)
ndim_x = len(train_data[0])-1
#print(np.shape(train_data))
meta_info = np.array([batch_size,ndim_x,predictor_num,n_replicas,file_addr])
np.savez(path + FILE_NAME, train_data, meta_info)
print('Predictor-%d dataset loaded from ' % predictor_num, file_addr,' and saved to '+FILE_NAME+'. Rows: %d ' % len(train_data[:,0,0]), ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)


""" import and create the train dataset into Numpy array  """
predictor_num = 2
FILE_NAME = 'traindata_p'+str(predictor_num)+'_'+str(int(batch_size/1000))+'k.npz'
training_dataset = ParquetDataset(file_addresses=file_addr,predictor_num=predictor_num)
train_data = training_dataset.get_data(batch_size,n_replicas)
ndim_x = len(train_data[0])-1
#print(np.shape(train_data))
meta_info = np.array([batch_size,ndim_x,predictor_num,n_replicas,file_addr])
np.savez(path + FILE_NAME, train_data, meta_info)
print('Predictor-%d dataset loaded from ' % predictor_num, file_addr,' and saved to '+FILE_NAME+'. Rows: %d ' % len(train_data[:,0,0]) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)


""" import and create the train dataset into Numpy array """
predictor_num = 3
FILE_NAME = 'traindata_p'+str(predictor_num)+'_'+str(int(batch_size/1000))+'k.npz'
training_dataset = ParquetDataset(file_addresses=file_addr,predictor_num=predictor_num)
train_data = training_dataset.get_data(batch_size,n_replicas)
ndim_x = len(train_data[0])-1
#print(np.shape(train_data))
meta_info = np.array([batch_size,ndim_x,predictor_num,n_replicas,file_addr])
np.savez(path + FILE_NAME, train_data, meta_info)
print('Predictor-%d dataset loaded from ' % predictor_num, file_addr,' and saved to '+FILE_NAME+'. Rows: %d ' % len(train_data[:,0,0]) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)
