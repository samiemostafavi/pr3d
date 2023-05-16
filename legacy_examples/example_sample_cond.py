# import tensorflow_probability as tfp
import time

import matplotlib.pyplot as plt
import numpy as np

# import pandas as pd
import seaborn as sns
import tensorflow as tf

from pr3d.de import (  # ConditionalGammaEVM,; ConditionalGammaMixtureEVM,
    ConditionalGaussianMM,
)

tf.config.set_visible_devices([], "GPU")

# import pyarrow as pa
# import pyarrow.parquet as pq


# from utils.dataset import create_dataset, load_parquet

dtype = "float64"  # 'float32' or 'float16'


"""
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
"""
N = 100000  # 489 seconds for 100k samples, 17 seconds for 10k samples
# plot conditional samples with x from the dataset
X = {
    "queue_length1": np.zeros(N),
    "queue_length2": np.zeros(N),
    "queue_length3": np.zeros(N),
}

# load the conditional trained model
conditional_delay_model = ConditionalGaussianMM(  # ConditionalGammaMixtureEVM, ConditionalGaussianMM
    h5_addr="gmm_conditional_model.h5",  # gmevm_conditional_model.h5, gmm_conditional_model.h5
    dtype=dtype,
)
start = time.time()
conditional_samples = conditional_delay_model.sample_n(
    x=X,
    rng=np.random.default_rng(12345),
)
end = time.time()
print(end - start)

fig, ax = plt.subplots()
sns.histplot(
    conditional_samples,
    kde=False,
    ax=ax,
)
ax.set_xlim(0, 100)
fig.tight_layout()
plt.savefig("cond_sample_n_test_gmevm.png")
