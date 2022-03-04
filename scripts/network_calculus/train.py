import warnings
import pickle
import sys
import os
import os.path
import math

warnings.filterwarnings('ignore')
ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
PRJ_PATH = os.path.dirname(__file__)
sys.path.append(ABS_PATH)

from cde.data_collector import ParquetDataset
from cde.density_estimator import NoNaNGPDExtremeValueMixtureDensityNetwork


""" Load or Create Project """
# this project uses the train dataset with arrival rate 0.95, and service rate 1

# Paths
data_path = PRJ_PATH+'/data/'
# create saves path
saves_path = PRJ_PATH+'/saves/'
dirname = os.path.dirname(saves_path)
os.makedirs(dirname, exist_ok=True)


""" import the training dataset into Numpy array """

file_addr = [data_path+'training_data_60k_1hop.parquet']
#dataset = ParquetDataset(file_addresses=file_addr,read_columns=['end2enddelay','h1_uplink_netstate'])
dataset = ParquetDataset(file_addresses=file_addr,read_columns=['totaldelay_downlink','h3_downlink_netstate'])
batch_size_train = dataset.n_records
train_data = dataset.get_data_leg(batch_size_train)
ndim_x = len(train_data[0])-1
#print(np.shape(train_data))
print('Predictor dataset loaded from ', file_addr,'. Rows: %d ' % len(train_data[:,0]), ' Columns: %d ' % len(train_data[0,:]), ' ndim_x: %d' % ndim_x)


""" create the model and train it """

n_epoch_emm = 3000
FILE_NAME = 'model_onehop_60k_overfit.pkl'

if os.path.isfile(saves_path + FILE_NAME):
    print('A trained model already exist.')
else:
    Y = train_data[:,0]
    X = train_data[:,1:]

    emm_model = NoNaNGPDExtremeValueMixtureDensityNetwork("EMM_A1", ndim_x=ndim_x, ndim_y=1, n_training_epochs=n_epoch_emm, n_centers=3, hidden_sizes=(16,16),verbose_step=math.floor(n_epoch_emm/10), weight_decay=0.0, learning_rate=2e-3,epsilon=1e-6,l2_reg=0.0, l1_reg=0.0,dropout=0.0)

    emm_model.fit(X, Y)

    with open(saves_path + FILE_NAME, 'wb') as output:
        pickle.dump(emm_model, output, pickle.HIGHEST_PROTOCOL)
