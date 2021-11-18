import time
import numpy as np
import sys

sys.path.insert(0, '../..')
from cde.data_collector import MatlabDataset, MatlabDatasetH5, get_most_common_unique_states

""" import the test dataset into Numpy array """

FILE_NAME = 'testdata_gamma.npz'
projects_path = 'saves/projects/'

file_addrs = ['../../experiments/data/package/records_A1.mat','../../experiments/data/package/records_A2.mat',
                '../../data/experiments/package/records_A3.mat','../../experiments/data/package/records_A4.mat',
                '../../data/experiments/package/records_B1.mat','../../experiments/data/package/records_B2.mat',
                '../../data/experiments/package/records_B3.mat','../../experiments/data/package/records_B4.mat']

content_keys = ['records_A1','records_A2','records_A3','records_A4',
                    'records_B1','records_B2','records_B3','records_B4']

select_cols = [0,1,5,9]
# define the test packet stream

start_time = time.time()
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