import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from pr3d.de import (  # ConditionalGammaEVM,
    ConditionalGammaMixtureEVM,
    ConditionalGaussianMM,
)

# import tensorflow_probability as tfp


dtype = "float64"  # 'float32' or 'float16'

N = 1000000
x = {"queue_length": np.ones(N) * 9, "longer_delay_prob": np.zeros(N)}
# x = { 'queue_length1': np.array([0, 10]), 'queue_length2': np.array([0, 10]), 'queue_length3' : np.array([0, 10]) }
x_list = np.array([np.array([*items]) for items in zip(*x.values())])

# load the conditional trained model
conditional_delay_model = ConditionalGammaMixtureEVM(
    h5_addr="test/gmevm_conditional_model_2.h5",
    dtype=dtype,
)

start = time.time()
samples = conditional_delay_model.sample_n(x=x)
end = time.time()
print(end - start)


fig, ax = plt.subplots()
sns.histplot(
    samples,
    kde=True,
    ax=ax,
    stat="density",
).set(title="count={0}".format(len(samples)))
ax.title.set_size(10)

# plot predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y0 = 0
y1 = 300
ax.set_xlim(y0, y1)
y_lin = np.linspace(y0, y1, N, dtype=dtype)
pdf, log_pdf, ecdf = conditional_delay_model.prob_batch(x=x_list, y=y_lin[:, None])
ax.plot(y_lin, pdf, "r", lw=2, label="prediction")
ax.legend()

plt.savefig("foo3.png")


# load the conditional trained model
conditional_delay_model = ConditionalGaussianMM(
    h5_addr="test/gmm_conditional_model_2.h5",
    dtype=dtype,
)

start = time.time()
samples = conditional_delay_model.sample_n(x=x)
end = time.time()
print(end - start)

fig, ax = plt.subplots()
sns.histplot(
    samples,
    kde=True,
    ax=ax,
    stat="density",
).set(title="count={0}".format(len(samples)))
ax.title.set_size(10)

# plot predictions
y0, y1 = ax.get_xlim()  # extract the y endpoints
y0 = 0
y1 = 300
ax.set_xlim(y0, y1)
y_lin = np.linspace(y0, y1, N, dtype=dtype)
pdf, log_pdf, ecdf = conditional_delay_model.prob_batch(x=x_list, y=y_lin[:, None])
ax.plot(y_lin, pdf, "r", lw=2, label="prediction")
ax.legend()

plt.savefig("foo2.png")

exit(0)
