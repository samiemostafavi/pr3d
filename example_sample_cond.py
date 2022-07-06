import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow_probability as tfp
import tensorflow as tf
import pyarrow as pa
import pyarrow.parquet as pq

from pr3d.de import ConditionalGammaEVM
from utils.dataset import create_dataset, load_parquet


dtype = 'float64' # 'float32' or 'float16'

# load dataset first
df = load_parquet(
    file_addresses=['dataset.parquet'],
    read_columns=None,
)
print(df)

# load the conditional trained model
conditional_delay_model = ConditionalGammaEVM(
    h5_addr = "evm_conditional_model.h5",
    dtype = dtype,
)


# load conditional data
conditional_df = df[
    (df.queue_length1<=1) & 
    (df.queue_length2>=1) & 
    (df.queue_length3>=4)
]

# plot conditional samples with x from the dataset
df = conditional_df[['queue_length1','queue_length2','queue_length3']]

X = ( df['queue_length1'].to_numpy(),df['queue_length2'].to_numpy(),df['queue_length3'].to_numpy() )
print(X)
conditional_samples = conditional_delay_model.sample_n(
    x = X,
    seed = 12345,
)