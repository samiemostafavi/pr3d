import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

from pr3d.de import GammaEVM, GaussianMM
from utils.dataset import create_dataset, load_parquet

dtype = 'float64' # 'float32' or 'float16'

df = load_parquet(file_addresses=['dataset_onehop.parquet'],read_columns=['service_delay'])
print(df)

# initiate the non conditional predictor
model = GaussianMM(
    #centers= 8,
    dtype = dtype,
    bayesian = True,
    #batch_size = 1024,
)

# shuffle the rows
training_df = df.sample(frac=1)

Y = training_df['service_delay'].to_numpy()

# train the model (it must be divisible by batch_size)
training_samples_num = 1024*10

# training
model.fit(
    Y[0:training_samples_num],
    batch_size = 1024, # 1024 training_samples_num
    epochs = 5000, # 5000
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
)

print(model.get_parameters())



model.save("bayes_gmm_service_delay_model.h5")
del model


df = load_parquet(file_addresses=['dataset_onehop.parquet'],read_columns=['service_delay'])

loaded_model = GaussianMM(
    h5_addr = "bayes_gmm_service_delay_model.h5"
)

print(f"Model loaded, bayesian: {loaded_model.bayesian}, batch_size: {loaded_model.batch_size}")
print(f"Parameters: {loaded_model.get_parameters()}")

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
if loaded_model.bayesian:
    for _ in range(100):
        pdf,log_pdf,ecdf = loaded_model.prob_batch(y = y_lin[:,None])
        ax.plot(y_lin, pdf, color='red', alpha=0.1, label='prediction')
else:
    pdf,log_pdf,ecdf = loaded_model.prob_batch(y = y_lin[:,None])
    ax.plot(y_lin, pdf, color='red', lw=2, label='prediction')

ax.legend(['Data', 'Prediction'])

fig.tight_layout()
plt.savefig('bayes_gmm_delay_fit.png')

