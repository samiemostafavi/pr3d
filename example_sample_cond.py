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

# load the conditional trained model
conditional_delay_model = ConditionalGammaEVM(
    h5_addr = "evm_conditional_model.h5",
    dtype = dtype,
)

# plot conditional samples with x from the dataset
X = { 'queue_length1': np.zeros(1000000), 'queue_length2': np.zeros(1000000), 'queue_length3' : np.zeros(1000000) }

conditional_samples = conditional_delay_model.sample_n(
    x = X,
    rng  = tf.random.Generator.from_seed(12345),
)

fig, ax = plt.subplots()
sns.histplot(
    conditional_samples,
    kde=False,
    ax = ax,
)
ax.set_xlim(0,100)
fig.tight_layout()
plt.savefig('cond_sample_n_test.png')