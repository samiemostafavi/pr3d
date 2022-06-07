import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from pr3d.de import GammaEVM
from .dataset import create_dataset, load_parquet

dtype = 'float64' # 'float32' or 'float16'

# initiate the predictor
evm_model = GammaEVM(
    dtype = dtype,
)

df = load_parquet(file_addresses=['dataset_onehop.parquet'],read_columns=['service_delay'])
print(df)

# shuffle the rows
training_df = df.sample(frac=1)

Y = training_df['service_delay'].to_numpy()

# train the model
training_samples_num = 10000
evm_model.fit(
    Y[1:training_samples_num],
    batch_size = training_samples_num, # 1000
    epochs = 5000, # 10
    learning_rate = 2e-2,
    weight_decay = 0.0,
    epsilon = 1e-8,
)

[gamma_shape,gamma_rate,gamma_rate,tail_param,tail_threshold,tail_scale] = evm_model.get_parameters()
print(gamma_shape,gamma_rate,gamma_rate,tail_param,tail_threshold,tail_scale)


fig, ax = plt.subplots()

sns.histplot(
    df,
    kde=True,
    ax = ax,
    stat="density",
).set(title="count={0}".format(len(df)))
ax.title.set_size(10)

# then, plot predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y_lin = np.linspace(y0, y1, 100, dtype=dtype)
pdf,log_pdf,ecdf = evm_model.prob_batch(y = y_lin[:,None])
ax.plot(y_lin, pdf, 'r', lw=2, label='prediction')
ax.legend()

fig.tight_layout()
plt.savefig('delay_fit.png')

evm_model.save("service_delay_model.h5")