import numpy as np

from pr3d.de import ConditionalGammaEVM, GammaEVM

from .utils.dataset import load_parquet

# load the fitted service delay model
dtype = "float64"
service_delay_model = GammaEVM(
    h5_addr="service_delay_model.h5",
)
print(
    "Service delay model is loaded. Parameters: {0}".format(
        service_delay_model.get_parameters()
    )
)

df = load_parquet(
    file_addresses=["dataset_onehop.parquet"],
    read_columns=None,
)
print(df)

# process the time_in_service and convert it to longer_delay_prob
tis = np.squeeze(df[["time_in_service"]].to_numpy())
longer_delay_prob = np.float64(1.00) - service_delay_model.prob_batch(tis)[2]
df["longer_delay_prob"] = longer_delay_prob
# replace NaNs with zeros
df["longer_delay_prob"] = df["longer_delay_prob"].fillna(np.float64(0.00))

print(df)
df.to_parquet("dataset_onehop_processed.parquet")

# train a conditional density network
# initiate the predictor
conditional_delay_model = ConditionalGammaEVM(
    x_dim=2,
    hidden_sizes=(16, 16, 16),
    dtype=dtype,
)

# separate X and Y and convert to numpy
X = df[["queue_length", "longer_delay_prob"]].to_numpy()
Y = df["end2end_delay"].to_numpy()

# train the model
training_samples_num = 50000
conditional_delay_model.fit(
    X[1:training_samples_num, :],
    Y[1:training_samples_num],
    batch_size=training_samples_num,  # 1000
    epochs=2000,  # 10
    learning_rate=1e-2,
    weight_decay=0.0,
    epsilon=1e-8,
)

conditional_delay_model.save("onehop_tis_model.h5")
