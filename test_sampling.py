import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pr3d.de import ConditionalGammaEVM, GammaEVM

from .utils.dataset import load_parquet

dtype = "float64"  # 'float32' or 'float16'

# initiate the predictor
service_delay_model = GammaEVM(
    h5_addr="service_delay_model.h5",
    dtype=dtype,
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

n = 2
m = 2
fig, axes = plt.subplots(nrows=n, ncols=m, figsize=(m * 4, n * 4))

ax = axes.flat[0]

sns.histplot(
    df["service_delay"],
    kde=True,
    ax=ax,
    stat="density",
).set(title="count={0}".format(len(df)))
ax.title.set_size(10)

# plot predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y_lin = np.linspace(y0, y1, 100, dtype=dtype)
pdf, log_pdf, ecdf = service_delay_model.prob_batch(y=y_lin[:, None])
ax.plot(y_lin, pdf, "r", lw=2, label="prediction")
ax.legend()

ax = axes.flat[1]

# plot samples
samples = service_delay_model.sample_n(
    1000,  # len(df),
    random_generator=np.random.default_rng(0),
)

sns.histplot(
    samples,
    kde=True,
    ax=ax,
    stat="density",
).set(title="count={0}".format(len(df)))
ax.title.set_size(10)


ax = axes.flat[2]

# load dataset first
df = load_parquet(
    file_addresses=["dataset_onehop_processed.parquet"],
    read_columns=None,
)
print(df)


# load the conditional trained model
conditional_delay_model = ConditionalGammaEVM(
    h5_addr="onehop_tis_model.h5",
    dtype=dtype,
)

values_count = (0,)
intervals = [0.8, 1]

# load conditional data
conditional_df = df[
    (df.queue_length == values_count[0])
    & (df.longer_delay_prob >= intervals[0])
    & (df.longer_delay_prob < intervals[1])
]
sns.histplot(conditional_df["end2end_delay"], kde=True, ax=ax, stat="density",).set(
    title="x={}, interval={}, count={}".format(
        values_count,
        ["{:0.2f}".format(inter) for inter in intervals],
        len(conditional_df),
    )
)
ax.title.set_size(10)


ax = axes.flat[3]

# plot conditional samples with x from the dataset
X = np.squeeze(conditional_df[["queue_length", "longer_delay_prob"]].to_numpy())
conditional_samples = conditional_delay_model.sample_n(
    x=X,
    random_generator=np.random.default_rng(0),
)

sns.histplot(
    conditional_samples,
    kde=True,
    ax=ax,
    stat="density",
).set(title="count={0}".format(len(conditional_df)))
ax.title.set_size(10)

fig.tight_layout()
plt.savefig("delay_fit_withsamples.png")

exit(0)
