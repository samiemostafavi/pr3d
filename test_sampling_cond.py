import numpy as np
from pr3d.de import ConditionalGammaEVM, ConditionalGaussianMM, ConditionalGammaMixtureEVM
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import time 

dtype = 'float64' # 'float32' or 'float16'

N = 1000000
x = { 'queue_length1': np.zeros(N), 'queue_length2': np.zeros(N), 'queue_length3' : np.zeros(N) }
#x = { 'queue_length1': np.array([0, 10]), 'queue_length2': np.array([0, 10]), 'queue_length3' : np.array([0, 10]) }
x_list = np.array([np.array([*items]) for items in zip(*x.values())])

# load the conditional trained model
conditional_delay_model = ConditionalGammaMixtureEVM(
    h5_addr = "test/gmevm_conditional_model.h5",
    dtype = dtype,
)

start = time.time()
samples = conditional_delay_model.sample_n(x=x)
end = time.time()
print(end-start)



fig, ax = plt.subplots()
sns.histplot(
    samples,
    kde=True,
    ax = ax,
    stat="density",
).set(title="count={0}".format(len(samples)))
ax.title.set_size(10)
plt.savefig('foo1.png')

exit(0)

# load the conditional trained model
conditional_delay_model = ConditionalGaussianMM(
    h5_addr = "test/gmm_conditional_model.h5",
    dtype = dtype,
)

start = time.time()
samples = conditional_delay_model.sample_n(x=x)
end = time.time()
print(end-start)

fig, ax = plt.subplots()
sns.histplot(
    samples,
    kde=True,
    ax = ax,
    stat="density",
).set(title="count={0}".format(len(samples)))
ax.title.set_size(10)
plt.savefig('foo2.png')