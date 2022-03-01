import warnings
import pickle
import matplotlib.pyplot as plt
import sys
import os
import warnings


warnings.filterwarnings('ignore')
ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PRJ_PATH = os.path.dirname(__file__)
sys.path.append(ABS_PATH)

from cde.data_collector import ParquetDataset
from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork
from cde.evaluation.empirical_eval import  evaluate_models_save_plots


# Paths
saves_path = PRJ_PATH+'/saves/'
data_path = PRJ_PATH+'/data/'

# create figures path
figures_path = PRJ_PATH+'/saves/figures/'
dirname = os.path.dirname(figures_path)
os.makedirs(dirname, exist_ok=True)

#most common: ([1,1,2],[1,1],[1])
network_state = [100]

""" import the test dataset into Numpy array """
file_addr = [data_path+'sim3hop_1_dataset_01_Feb_2022_10_20_42.parquet']
test_dataset = ParquetDataset(file_addresses=file_addr,read_columns=['end2enddelay','h1_uplink_netstate'])
test_data = test_dataset.get_data_unshuffled(test_dataset.n_records)
ndim_x_test = len(test_data[0])-1
#print(np.shape(train_data))
print('Test dataset loaded from ', file_addr,'. Rows: %d ' % len(test_data[:,0]), ' Columns: %d ' % len(test_data[0,:]), ' ndim_x: %d' % ndim_x_test)

""" import the training dataset into Numpy array """
file_addr = [data_path+'sim3hop_1_dataset_27_Feb_22k_correct_noised.parquet']
train_dataset = ParquetDataset(file_addresses=file_addr,read_columns=['end2enddelay','h1_uplink_netstate'])
train_data = train_dataset.get_data_unshuffled(train_dataset.n_records)
ndim_x_train = len(train_data[0])-1
print('Train dataset loaded from ', file_addr,'. Rows: %d ' % len(train_data[:,0]), ' Columns: %d ' % len(train_data[0,:]), ' ndim_x: %d' % ndim_x_train)

""" load trained emm model """
FILE_NAME = 'model_onehop_22k_correct_noised.pkl'
with open(saves_path + FILE_NAME, 'rb') as input:
    model = NoNaNGPDExtremeValueMixtureDensityNetwork(name='EMM'+str(4), ndim_x=1, ndim_y=1)
    model._setup_inference_and_initialize()
    model = pickle.load(input)
    print(model)

    print(model._get_tail_components([network_state]))
    print(model._get_mixture_components([network_state]))

    plt.style.use(PRJ_PATH+'/plot_style.txt')
    evaluate_models_save_plots(
        models=[model],
        model_names=["EMM prediction"],
        train_data=train_data,
        cond_state=network_state,
        test_dataset=test_data,
        quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-4,1-1e-5,1-1e-6],
        save_fig_addr=figures_path+'figure_',
        xlim = [0,14,0.001,14],
        loglog=False,
    )


#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[1],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])
#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[2],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])
#evaluate_models_singlestate(models=[model],model_names=["EMM"],train_data=train_data,cond_state=[10],test_dataset=test_data,quantiles=[1-1e-1,1-1e-2,1-1e-3,1-1e-5])

#unique_states,_,_ = get_most_common_unique_states(test_data[1000000:5000000,:],ndim_x=1,N=30,plot=True,save_fig_addr=path)
