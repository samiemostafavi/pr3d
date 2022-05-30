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

exit(0)

# convert the created dataset into pandas dataframe
df = pd.DataFrame(np.concatenate((Y[:, None],X), axis=1))
df.columns =['end2end_delay', 'queue_length1', 'queue_length2', 'queue_length3']
print(df)


# prepare matplot env
plt.figure()
sns.set_style('darkgrid')

# empirical kde density function
x_cond = np.array([0,0,0],dtype=dtype)
conditional_df = df[(df.queue_length1==x_cond[0]) & (df.queue_length2==x_cond[1]) & (df.queue_length3==x_cond[2]) ]
facet_grid = sns.displot(conditional_df['end2end_delay'],kde=True,stat="density")
ax = facet_grid.ax

# emm predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y_lin = np.linspace(y0, y1, 100, dtype=dtype)
x_lin = np.tile(x_cond, (len(y_lin),1))
pdf,log_pdf,ecdf = emm_model.prob_batch(x = x_lin, y = y_lin[:,None])
ax.plot(y_lin, pdf, 'r', lw=2, label='pdf')
ax.legend()
#print(y_lin)
#print(pdf)

x_cond = np.array([0,2,0],dtype=dtype)
conditional_df = df[(df.queue_length1==x_cond[0]) & (df.queue_length2==x_cond[1]) & (df.queue_length3==x_cond[2]) ]
facet_grid = sns.displot(conditional_df['end2end_delay'],kde=True,stat="density")
ax = facet_grid.ax

# emm predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y_lin = np.linspace(y0, y1, 100, dtype=dtype)
x_lin = np.tile(x_cond, (len(y_lin),1))
pdf,log_pdf,ecdf = emm_model.prob_batch(x = x_lin, y = y_lin[:,None])
ax.plot(y_lin, pdf, 'r', lw=2, label='pdf')
ax.legend()
#print(y_lin)
#print(pdf)

plt.show()