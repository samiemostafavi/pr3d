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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.density_estimator import MixtureDensityNetwork

# State and budget 

xs = ([1,4,2], [4,2],    [2])
nus = (20,     18.34,    14.02)

# Path
path = '../training/saves/'

predictor_nums = [1,2,3]
model_strs = ['emm','gmm']
batch_sizes = [10000]
replica_nums = [0]

for predictor_num in predictor_nums:
    for model_str in model_strs:
        for batch_size in batch_sizes:
            for replica_num in replica_nums:
                """ load trained emm models """
                FILE_NAME = 'trained_'+model_str+'_p'+str(predictor_num)+'_s'+str(int(batch_size/1000))+'_r'+str(replica_num)+'.pkl'
                if not os.path.isfile(path + 'trained_models/' + FILE_NAME):
                    print('No trained model found.')
                    exit()
                with open(path + 'trained_models/' + FILE_NAME, 'rb') as input:
                    if model_str is 'emm':
                        model = NoNaNGPDExtremeValueMixtureDensityNetwork(name=model_str+str(predictor_num), ndim_x=4-predictor_num, ndim_y=1)
                    else:
                        model = MixtureDensityNetwork(name=model_str+str(predictor_num), ndim_x=4-predictor_num, ndim_y=1)
                    model._setup_inference_and_initialize()
                    model = pickle.load(input)

                print(model_str + '-p' + str(predictor_num) + ', batch_size=' + str(int(batch_size/1000)) + ', replica=' + str(replica_num) + ', x=' + str(xs[predictor_num-1]) + ', nu=' + str(nus[predictor_num-1]) + ', DVP: ' + str(model.tail(np.array([xs[predictor_num-1]]),np.array([nus[predictor_num-1]]))))

