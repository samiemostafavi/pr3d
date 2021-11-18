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


""" Load the benchamrk results of the models upon all the states """

file_names = ['eval_onehop_dual_p95_5k.npz','eval_onehop_dual_p95_10k.npz','eval_onehop_dual_p95_20k.npz']
titles = ['5k','10k','20k']

npzfile = np.load(path + file_names[0])
meta_info = npzfile['arr_0']
quantiles = npzfile['arr_1']
xsize = int(meta_info[0])
N_us = int(meta_info[1])
eval_results = np.empty((0,N_us,xsize+(len(quantiles))+1))
agg_tails = []
agg_titles = []

for i in range(len(file_names)):

    npzfile = np.load(path + file_names[i])
    meta_info = npzfile['arr_0']
    quantiles = npzfile['arr_1']
    xsize = int(meta_info[0])
    N_us = int(meta_info[1])
    len_train_data = int(meta_info[2])
    n_epoch_emm = int(meta_info[3])

    emm_results_p = npzfile['arr_6']
    eval_results = np.append(eval_results,[emm_results_p],axis=0)
    agg_titles.append('EMM ' + titles[i])

    emm_results_p = npzfile['arr_7']
    eval_results = np.append(eval_results,[emm_results_p],axis=0)
    agg_titles.append('GMM ' + titles[i])

    tail_results = npzfile['arr_3']
    tail_x_axis = npzfile['arr_4']
    tail_x_quants = npzfile['arr_5']

    agg_tails.append(tail_results[0])
    agg_tails.append(tail_results[1])

    print('benchmark allstate loaded from .npz file. Models: %d ' % len(eval_results), ' n_states(N): %d ' % len(eval_results[0]), 'xsize: %d' % xsize)


# reorder
eval_order = [4, 5, 2, 3, 0, 1]
eval_results = eval_results[eval_order,:]
eval_titles = [agg_titles[i] for i in eval_order]

tail_order = [0, 1, 2, 3, 4, 5]
agg_tails = [agg_tails[i] for i in tail_order]
tail_titles = [agg_titles[i] for i in tail_order]


plt.style.use('plot_style.txt')
evaluate_models_allstates_plot_save(cma_results=eval_results,train_len=len_train_data,model_names=eval_titles,quantiles=quantiles,xsize=xsize,markers=['s','o','v','*','x'],loglog=True,ylim_ll=[1e-4,1e2],marker_size=80,save_fig_addr=path+'eval_aio_')
#evaluate_models_allstates_plot_save(cma_results=eval_results,train_len=len_train_data,model_names=['EMM'],quantiles=quantiles,xsize=xsize,markers=['s','o','v','*','x'],loglog=False,ylim=[0,100],marker_size=80,save_fig_addr=path+'eval2_')
evaluate_models_allstates_agg_save(cma_results=eval_results,train_len=len_train_data,n_epoch=n_epoch_emm,quantiles=quantiles,xsize=xsize,model_names=eval_titles,ebar_width=0.7,save_fig_addr=path+'eval_aio_')

""" Load and print the tail estimation errors """

plt.style.use('plot_style.txt')
evaluate_models_tail_agg_plot_save(agg_tails,tail_x_axis,tail_x_quants,model_names=tail_titles,plotylim=[5,0],save_fig_addr=path+'eval_aio_')
