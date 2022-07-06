import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

from pr3d.de import ConditionalGammaEVM, ConditionalGaussianMM
from utils import parquet_tf_pipeline_2

dtype = 'float64' # 'float32' or 'float16'

dataset_size = 8192 # = 1024*8, max 8995
train_size = 7168 #int(dataset_size * 0.85)
batch_size = 1024 #256

feature_names = [
    "queue_length1",
    "queue_length2",
    "queue_length3",
]

label_name = "end2end_delay"

train_dataset, test_dataset = parquet_tf_pipeline_2(
    file_addr = './dataset.parquet',
    feature_names = feature_names,
    label_name = label_name,
    dataset_size = dataset_size,
    train_size = train_size,
    batch_size = batch_size,
)

#print(train_dataset)
#for ds in train_dataset:
#    print(ds)

#print(test_dataset)
#for ds in test_dataset:
#    print(ds)

# initiate the non conditional predictor
model = ConditionalGammaEVM(
    #centers= 4,
    bayesian = False,
    #batch_size = 1024,
    x_dim = feature_names,
    hidden_sizes = (16,16), #(20, 50, 20) for bayesian, (16,16) for non-bayesian
    hidden_activation = 'tanh', #'sigmoid'
    dtype = dtype,
)

model.fit_pipeline(
    train_dataset,
    test_dataset,
    optimizer = keras.optimizers.Adam(learning_rate=0.01),
    batch_size = 1024,
    epochs = 200, #1000
)

condition = [0,0,0]

print(f"Parameters of {condition}: {model.get_parameters(x=condition)}")

#model.core_model.model.summary()

model.save("evm_conditional_model.h5")
del model


loaded_model = ConditionalGammaEVM(
    h5_addr = "evm_conditional_model.h5"
)

print(f"Model loaded, bayesian: {loaded_model.bayesian}, x_dim: {loaded_model.x_dim}, hidden_sizes: {loaded_model.hidden_sizes}")
print(f"Parameters: {loaded_model.get_parameters(x=condition)}")


# load dataset first
file_addresses = ['dataset.parquet']
table = pa.concat_tables(
    pq.read_table(
        file_address,columns=None,
    ) for file_address in file_addresses
)
df = table.to_pandas()
print(df)

conditional_df = df[
    (df.queue_length1==condition[0]) & 
    (df.queue_length2==condition[1]) & 
    (df.queue_length3==condition[2])
]
fig, ax = plt.subplots()
sns.histplot(
    conditional_df['end2end_delay'],
    kde=True,
    ax = ax,
    stat="density",
).set(title="x={0}, count={1}".format(condition,len(conditional_df)))
ax.title.set_size(10)

# then, plot predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y_lin = np.linspace(y0, y1, 100, dtype=dtype)
x_lin = np.tile(condition, (len(y_lin),1))
if loaded_model.bayesian:
    for _ in range(100):
        pdf,log_pdf,ecdf = loaded_model.prob_batch(x = x_lin, y = y_lin[:,None])
        ax.plot(y_lin, pdf, color='red', alpha=0.1, label='prediction')
else:
    pdf,log_pdf,ecdf = loaded_model.prob_batch(x = x_lin, y = y_lin[:,None])
    ax.plot(y_lin, pdf, color='red', lw=2, label='prediction')

ax.legend(['Data', 'Prediction'])

fig.tight_layout()
plt.savefig('evm_conditional_model.png')
