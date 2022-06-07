import numpy as np

from pr3d.de import ConditionalGaussianMM
from .dataset import create_dataset

dtype = 'float64' # 'float32' or 'float16'

gmm_model = ConditionalGaussianMM(
    centers = 8,
    x_dim = 3,
    hidden_sizes = (16,16),
    dtype = dtype,
)

np.random.seed(0)
X,Y = create_dataset(n_samples = 10000, x_dim = 3, dtype = dtype)
print("X shape: {0}".format(X.shape))
print("Y shape: {0}".format(Y.shape))

# train the model
gmm_model.fit(
    X,Y,
    batch_size = 10000, # 1000
    epochs = 1000, # 10
    learning_rate = 5e-2,
    weight_decay = 0.0,
    epsilon = 1e-8,
)

print("Single test x: {0}, and y: {1}".format(X[10,:],Y[10]))
pdf,log_pdf,ecdf = gmm_model.prob_single(X[10,:],Y[10])
print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

print("Batch test x: {0}, and y: {1}".format(X[10:15,:],Y[10:15]))
pdf,log_pdf,ecdf = gmm_model.prob_batch(X[10:15,:],Y[10:15])
print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

gmm_model.save("gmm_model.h5")
