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


from ..cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from ..cde.density_estimator import MixtureDensityNetwork

from ..cde.data_collector import MatlabDataset, MatlabDatasetH5, get_most_common_unique_states
from ..cde.density_estimator import plot_conditional_hist, measure_percentile, measure_percentile_allsame, measure_tail, measure_tail_allsame, init_tail_index_hill, estimate_tail_index_hill
from ..cde.evaluation.empirical_eval import evaluate_models_singlestate, empirical_measurer, evaluate_model_allstates, evaluate_models_allstates_plot, obtain_exp_value, evaluate_models_allstates_agg


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

FILE_NAME = 'traindata_1hop_p95_20k.npz'

try:
    npzfile = np.load(path + FILE_NAME)
    train_data = npzfile['arr_0']
    meta_info = npzfile['arr_1']
    batch_size = meta_info[0]
    ndim_x = meta_info[1]

    print('training data loaded from .npz file. Rows: %d ' % len(train_data), ' Columns: %d ' % len(train_data[0]), 'Batch_size: %d' % batch_size, 'ndim_x: %d' % ndim_x)
except:
    print('train data .npz file does not exist, import and create the train dataset into Numpy array')

    """ import and create the train dataset into Numpy array """

    file_addr = '../../data/train_records_single_p95.mat'
    content_key = 'train_records'
    select_cols = [0,1]
    batch_size = 20000

    training_dataset = MatlabDataset(file_address=file_addr,content_key=content_key,select_cols=select_cols)
    train_data = training_dataset.get_data(batch_size)
    ndim_x = len(train_data[0])-1
    meta_info = np.array([batch_size,ndim_x])
    np.savez(path + FILE_NAME, train_data, meta_info)

    print('train data loaded from .mat files. Rows: %d ' % len(train_data), ' Columns: %d ' % len(train_data[0]), 'ndim_x: %d' % ndim_x)


""" create the model and train it """

n_epoch_gmm = 12000
FILE_NAME = 'gmm_onehop_p95_20k.pkl'

if os.path.isfile(path + FILE_NAME):
    print('A trained model already exist.')
else:
    Y = train_data[:,0]
    X = train_data[:,1:]

    gmm_model = MixtureDensityNetwork("GMM_A", ndim_x=ndim_x, n_centers=3, ndim_y=1,n_training_epochs=n_epoch_gmm,hidden_sizes=(16, 16))
    gmm_model.fit(X, Y)

    with open(path + FILE_NAME, 'wb') as output:
        pickle.dump(gmm_model, output, pickle.HIGHEST_PROTOCOL)

n_epoch_emm = 30000
FILE_NAME = 'model_onehop_p95_20k.pkl'

if os.path.isfile(path + FILE_NAME):
    print('A trained model already exist.')
else:
    Y = train_data[:,0]
    X = train_data[:,1:]

    emm_model = NoNaNGPDExtremeValueMixtureDensityNetwork("EMM_A33", ndim_x=ndim_x, n_centers=2, ndim_y=1, n_training_epochs=n_epoch_emm, hidden_sizes=(16, 16),verbose_step=math.floor(n_epoch_emm/10), weight_decay=0, learning_rate=1e-3,epsilon=1e-6,l2_reg=0.0, l1_reg=0.0,dropout=0.0)
    emm_model.fit(X, Y)

    with open(path + FILE_NAME, 'wb') as output:
        pickle.dump(emm_model, output, pickle.HIGHEST_PROTOCOL)
