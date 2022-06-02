import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from cde import ConditionalEMM
from cde import create_dataset, load_parquet

dtype = 'float64' # 'float32' or 'float16'

# initiate the predictor
emm_model = ConditionalEMM(
    centers = 2,
    x_dim = 3,
    hidden_sizes = (16,16),
    dtype = dtype,
)

#np.random.seed(0)
#X,Y = create_dataset(n_samples = 50000, x_dim = 3, x_max = 5, x_level=1, dtype = dtype, dist = 'gamma')
#print("X shape: {0}".format(X.shape))
#print("Y shape: {0}".format(Y.shape))

df = load_parquet(file_addresses=['dataset.parquet'],read_columns=['end2end_delay','queue_length1','queue_length2','queue_length3'])
print(df)

# shuffle the rows
training_df = df.sample(frac=1)

# separate X and Y and convert to numpy
X = training_df[['queue_length1','queue_length2','queue_length3']].to_numpy()
Y = training_df['end2end_delay'].to_numpy()

# train the model
training_samples_num = 9000
emm_model.fit(
    X[1:training_samples_num,:],Y[1:training_samples_num],
    batch_size = training_samples_num, # 1000
    epochs = 5000, # 10
    learning_rate = 1e-2,
    weight_decay = 0.0,
    epsilon = 1e-8,
)

emm_model.save("emm_model.h5")
