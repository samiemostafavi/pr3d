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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cde.data_collector import ParquetDataset
from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork
from cde.evaluation.empirical_eval import  evaluate_models_save_plots


# Path
path_train = '../training/saves/'
path_valid = 'saves/'

warnings.filterwarnings('ignore')

predictor_nums = [1,2,3]
model_strs = ['emm','gmm']
batch_sizes = [10000]
replica_nums = [0]

#most common: ([1,1,2],[1,1],[1])
network_states = ([1,4,2],[4,2],[2])

models = []
for idx,predictor_num in enumerate(predictor_nums):

    """ import the test dataset into Numpy array """
    file_addr = ['../data/sim3hop_1_dataset_06_Sep_2021_11_20_40.parquet']
    batch_size_test = 80000000
    test_dataset = ParquetDataset(file_addresses=file_addr,predictor_num=predictor_num)
    test_data = test_dataset.get_data_unshuffled(batch_size_test)
    ndim_x_test = len(test_data[0])-1
    #print(np.shape(train_data))
    print('Predictor-%d dataset loaded from ' % predictor_num, file_addr,'. Rows: %d ' % len(test_data[:,0]), ' Columns: %d ' % len(test_data[0,:]), ' ndim_x: %d' % ndim_x_test)

    models_type = []
    for batch_size in batch_sizes:

        """ load training data """
        FILE_NAME = 'traindata_p'+str(predictor_num)+'_'+str(int(batch_size/1000))+'k.npz'
        npzfile = np.load(path_train + FILE_NAME)
        train_data = npzfile['arr_0']
        meta_info = npzfile['arr_1']
        batch_size = int(meta_info[0])
        ndim_x = int(meta_info[1])
        predictor_num = int(meta_info[2])
        n_replicas = int(meta_info[3])
        print('Predictor-%d training data loaded from .npz file. Rows: %d ' %(predictor_num,len(train_data[:,0,0])) , 'with Batch size: %dk ' % int(batch_size/1000) , ' Columns: %d ' % len(train_data[0,:,0]), ' Replicas: %d' % len(train_data[0,0,:]) , ' ndim_x: %d' % ndim_x)

        # currently only one batch size so no model array
        for replica_num in replica_nums:
            train_data = train_data[:,:,replica_num]

            for model_str in model_strs:
                """ load trained emm models """
                FILE_NAME = 'trained_'+model_str+'_p'+str(predictor_num)+'_s'+str(int(batch_size/1000))+'_r'+str(replica_num)+'.pkl'
                with open(path_train + 'trained_models/' + FILE_NAME, 'rb') as input:
                    if model_str is 'emm':
                        model = NoNaNGPDExtremeValueMixtureDensityNetwork(name=model_str+str(predictor_num), ndim_x=4-predictor_num, ndim_y=1)
                    else:
                        model = MixtureDensityNetwork(name=model_str+str(predictor_num), ndim_x=4-predictor_num, ndim_y=1)
                    model._setup_inference_and_initialize()
                    model = pickle.load(input)
                    models_type.append(model)
                    print(model)

        
        network_state = network_states[predictor_num-1]
        plt.style.use('plot_style.txt')
        evaluate_models_save_plots(models=models_type,model_names=["EMM prediction","GMM prediction"],train_data=train_data,cond_state=network_state,test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-4,1-1e-5],save_fig_addr=path_valid+'dual_test_')

    models.append(models_type)

print('models shape: ' + str([len(models),len(models[0])]))

#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[1],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])
#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[2],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])
#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[10],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])

#unique_states,_,_ = get_most_common_unique_states(test_data[1000000:5000000,:],ndim_x=1,N=30,plot=True,save_fig_addr=path)
