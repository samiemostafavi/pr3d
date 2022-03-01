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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork

from cde.data_collector import MatlabDataset, MatlabDatasetH5, get_most_common_unique_states
from cde.density_estimator import plot_conditional_hist, measure_percentile, measure_percentile_allsame, measure_tail, measure_tail_allsame, init_tail_index_hill, estimate_tail_index_hill
from cde.evaluation.empirical_eval import evaluate_models_singlestate, empirical_measurer, evaluate_model_allstates, evaluate_models_allstates_plot, obtain_exp_value, evaluate_models_allstates_agg, evaluate_models_save_plots, evaluate_models_allstates_plot_save, evaluate_models_allstates_agg_save,evaluate_models_tail_agg_plot_save


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
        cond_matds = MatlabDataset(file_address=file_addrs[i],content_key=content_keys[i],select_cols=select_cols)
        if i is 0:
            test_data = cond_matds.dataset
        else:
            test_data = np.append(test_data,cond_matds.dataset,axis=0)

    ndim_x_test = len(test_data[0])-1

    meta_info = np.array([ndim_x_test])
    np.savez(projects_path + FILE_NAME, test_data, meta_info)

    print('test data loaded from .mat files. Rows: %d ' % len(test_data), ' Columns: %d ' % len(test_data[0]), 'ndim_x: %d' % ndim_x_test)

ndim_x = ndim_x_test

""" Load the benchamrk results of the models upon all the states """

FILE_NAME = 'eval_onehop_p95.npz'

npzfile = np.load(path + FILE_NAME)
meta_info = npzfile['arr_0']
quantiles = npzfile['arr_1']
xsize = int(meta_info[0])
N_us = int(meta_info[1])
len_train_data = int(meta_info[2])
n_epoch_emm = int(meta_info[3])

emp_model = empirical_measurer(dataset=test_data,xsize=xsize,quantiles=quantiles)
emp_model.database = npzfile['arr_2']

eval_results = np.empty((0,N_us,xsize+(len(quantiles))+1))
emm_results_p = npzfile['arr_3']
eval_results = np.append(eval_results,[emm_results_p],axis=0)

print('benchmark allstate loaded from .npz file. Models: %d ' % len(eval_results), ' n_states(N): %d ' % len(eval_results[0]), 'xsize: %d' % xsize)

plt.style.use('plot_style.txt')
evaluate_models_allstates_plot_save(cma_results=eval_results,train_len=len_train_data,model_names=['EMM'],quantiles=quantiles,xsize=xsize,markers=['s','o','v','*','x'],loglog=True,ylim_ll=[1e-4,1e2],marker_size=80,save_fig_addr=path+'eval2_')
#evaluate_models_allstates_plot_save(cma_results=eval_results,train_len=len_train_data,model_names=['EMM'],quantiles=quantiles,xsize=xsize,markers=['s','o','v','*','x'],loglog=False,ylim=[0,100],marker_size=80,save_fig_addr=path+'eval2_')
evaluate_models_allstates_agg_save(cma_results=eval_results,train_len=len_train_data,n_epoch=n_epoch_emm,quantiles=quantiles,xsize=xsize,model_names=['EMM'],save_fig_addr=path+'eval2_')

""" Load and print the tail estimation errors """

tail_results = npzfile['arr_4']
tail_x_axis = npzfile['arr_5']
tail_x_quants = npzfile['arr_6']

plt.style.use('plot_style.txt')
evaluate_models_tail_agg_plot_save(tail_results,tail_x_axis,tail_x_quants,model_names=['EMM'],save_fig_addr=path)
