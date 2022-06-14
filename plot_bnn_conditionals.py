import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from keras import layers
import scipy.stats as sc

from bayesian import load_bnn_model

# load dataset first
file_addresses = ['dataset.parquet']
table = pa.concat_tables(
    pq.read_table(
        file_address,columns=None,
    ) for file_address in file_addresses
)
df = table.to_pandas()
print(df)

# load the trained model
dtype = 'float32'

h5_addr = "bnn_threehop_model.h5"
conditional_delay_model = load_bnn_model(h5_addr)

# find 10 most common queue_length occurances
n = 6; n1 = 3; n2 = 2;
x_lim = [20,100]

#values_count = df[['queue_length1','queue_length2','queue_length3']].value_counts()[:n].index.tolist()
values_count = [(0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (2, 2, 3), (5, 5, 5)]
print("{0} most common queue states: {1}".format(n,values_count))
#values_count = [(1,1,1),(0,0,0),(0,5,0),(0,10,0)]

# plot the conditional distributions of them
fig, axes = plt.subplots(nrows=n1, ncols=n2, figsize=(n1*4,n2*5))

for i, ax in zip(range(n), axes.flat):
    # first, plot empirical histogram
    conditional_df = df[
        (df.queue_length1==values_count[i][0]) & 
        (df.queue_length2==values_count[i][1]) & 
        (df.queue_length3==values_count[i][2])
    ]
    sns.histplot(
        conditional_df['end2end_delay'],
        kde=True, 
        ax=ax,
        stat="density",
    ).set(title="x={0}, count={1}".format(values_count[i],len(conditional_df)))
    ax.title.set_size(10)
    ax.set_xlim(left=x_lim[0], right=x_lim[1])

    # then, plot predictions
    x_cond = [ 
        np.array([values_count[i][0]]),
        np.array([values_count[i][1]]),
        np.array([values_count[i][2]]),
    ]
    y0, y1 = ax.get_xlim()  # extract the y endpoints
    y_lin = np.linspace(y0, y1, 100, dtype=dtype)
    x_lin = np.tile(x_cond, (len(y_lin),1))

    for _ in range(100):
        # sample the BNN with x (conditions) from the empirical data
        prediction_distribution = conditional_delay_model(x_cond)
        pdf = sc.norm(
            np.squeeze(prediction_distribution.mean()), 
            np.squeeze(prediction_distribution.stddev())).pdf(y_lin)
        ax.plot(y_lin, pdf, color='red', alpha=0.1)
        #ax.plot(y_lin, pdf, 'r', lw=2, label='prediction')
        #ax.legend()

    ax.legend(['Data', 'Prediction'])

    #for _ in range(10):

    #print(prediction_distributions.mean())
    #print(prediction_distributions.stddev())
    #prediction_distributions = conditional_delay_model([X1,X2])
    #print(prediction_distributions.mean())
    #print(prediction_distributions.stddev())
    #prediction_distributions = conditional_delay_model([X1,X2])
    #print(prediction_distributions.mean())
    #print(prediction_distributions.stddev())

fig.tight_layout()
plt.savefig('bnn_conditional_delay.png')