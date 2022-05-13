import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from cde import ConditionalEMM
from cde import create_dataset

dtype = 'float64' # 'float32' or 'float16'

emm_model = ConditionalEMM(
    centers = 2,
    x_dim = 3,
    hidden_sizes = (16,16),
    dtype = dtype,
)

np.random.seed(0)
X,Y = create_dataset(n_samples = 10000, x_dim = 3, dtype = dtype)
print("X shape: {0}".format(X.shape))
print("Y shape: {0}".format(Y.shape))

# train the model
"""
emm_model.fit(
    X,Y,
    batch_size = 10000, # 1000
    epochs = 1000, # 10
    learning_rate = 5e-2,
    weight_decay = 0.0,
    epsilon = 1e-8,
)
"""

# Process the collected data
df = pd.DataFrame(np.concatenate((Y,X), axis=0))