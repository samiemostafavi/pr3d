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
import pandas as pd

print(tf.__version__) #1.7.0
print(pd.__version__) #0.24 / 0.25.3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork

from cde.data_collector import MatlabDataset, MatlabDatasetH5, get_most_common_unique_states
from cde.density_estimator import plot_conditional_hist, measure_percentile, measure_percentile_allsame, measure_tail, measure_tail_allsame, init_tail_index_hill, estimate_tail_index_hill
from cde.evaluation.empirical_eval import evaluate_models_singlestate, empirical_measurer, evaluate_model_allstates, evaluate_models_allstates_plot, obtain_exp_value, evaluate_models_allstates_agg, evaluate_models_save_plots,evaluate_models_allstates_plot_save, evaluate_model_allstates_tail


""" Load or Create Project """
# this project uses the train dataset with arrival rate 0.95, and service rate 1

# name
PROJECT_NAME = 'onehop_p95'

# Path
projects_path = 'saves/projects/'
path = projects_path + PROJECT_NAME + '/'
  
# Get the directory name from the specified path
dirname = os.path.dirname(path)

# create the dir
os.makedirs(dirname, exist_ok=True)



""" load training data """

FILE_NAME = 'traindata_1hop_p95_1k.npz'

try:
    npzfile = np.load(path + FILE_NAME)
    train_data = npzfile['arr_0']
    meta_info = npzfile['arr_1']
    batch_size = meta_info[0]
    ndim_x = meta_info[1]

    print('training data loaded from .npz file. Rows: %d ' % len(train_data), ' Columns: %d ' % len(train_data[0]), 'Batch_size: %d' % batch_size, 'ndim_x: %d' % ndim_x)
except:
    print('train data .npz file does not exist!')
    exit()



""" import the test dataset into Numpy array """

FILE_NAME = 'testdata_onehop_p95.npz'

try:

    npzfile = np.load(projects_path + FILE_NAME)
    test_data = npzfile['arr_0']
    meta_info = npzfile['arr_1']
    ndim_x_test = meta_info[0]

    print('test data loaded from .npz file. Rows: %d ' % len(test_data), ' Columns: %d ' % len(test_data[0]), 'ndim_x: %d' % ndim_x_test)
except:
    print('test data .npz file does not exist, import and create the test dataset into Numpy array')

    file_addrs = ['../../data/onehop_records_p95.mat']
    content_keys = ['records']
    select_cols = [0,1]
    # define the test packet stream

    # 130 seconds each, 15 minutes total
    for i in range(len(file_addrs)):
        cond_matds = MatlabDatasetH5(file_address=file_addrs[i],content_key=content_keys[i],select_cols=select_cols)
        if i is 0:
            test_data = cond_matds.dataset
        else:
            test_data = np.append(test_data,cond_matds.dataset,axis=0)

    ndim_x_test = len(test_data[0])-1

    meta_info = np.array([ndim_x_test])
    np.savez(projects_path + FILE_NAME, test_data, meta_info)

    print('test data loaded from .mat files. Rows: %d ' % len(test_data), ' Columns: %d ' % len(test_data[0]), 'ndim_x: %d' % ndim_x_test)

ndim_x = ndim_x_test

""" load trained models """

FILE_NAME = 'model_onehop_p95_1k_2.pkl'
if not os.path.isfile(path + FILE_NAME):
    print('No trained model found.')
    exit()

with open(path+FILE_NAME, 'rb') as input:
    model = NoNaNGPDExtremeValueMixtureDensityNetwork(name="", ndim_x=ndim_x, ndim_y=1)
    model._setup_inference_and_initialize()
    model = pickle.load(input)

n_epoch = model.n_training_epochs

print(model)
print('n_epoch: %d'%n_epoch)


""" Find most common states in the test_data """

N_us=20 # N=60
unique_states,_,_ = get_most_common_unique_states(test_data[:900000,:],ndim_x=1,N=N_us,plot=True,save_fig_addr=path+'eval_')


""" Benchamrk the models upon all the states """
""" Requirements: models, emp_model, test_data, ndim_x_test, and unique_states """

FILE_NAME = 'eval_onehop_p95.npz'

quantiles = [0.9, 0.99, 0.999, 0.9999, 0.99999] #quantiles = [0.9, 0.99, 0.999, 0.9999, 0.99999]
xsize = ndim_x_test

emp_model = empirical_measurer(dataset=test_data,xsize=xsize,quantiles=quantiles)

# warning - takes 1 hour
# the first model treats with .tail_inverse, the rest with find_perc
# N=30, len(quantiles)=5, 1.5 hour

eval_results = np.empty((0,N_us,xsize+(len(quantiles))+1))

# EMM
results_1 = evaluate_model_allstates(emp_model=emp_model,model=model,train_data=train_data,unique_states=unique_states,N=N_us,quantiles=quantiles,xsize=xsize,root_find=False)
eval_results = np.append(eval_results,[results_1],axis=0)


# Tail prediction evaluation
N_yt=15
ylimT = [0.00001, 0.99999]
tail_results,tail_x_axis,tail_x_quants = evaluate_model_allstates_tail([model],test_data,unique_states,N=N_us,N_y=N_yt,ylim=ylimT,xsize=xsize)


# save the results into a file
meta_info = np.array([xsize, N_us, len(train_data), n_epoch, N_yt, ylimT[0], ylimT[1]])
quantiles_info = np.array(quantiles)
np.savez(path + FILE_NAME, meta_info, quantiles_info, emp_model.database, results_1, tail_results, tail_x_axis, tail_x_quants)
