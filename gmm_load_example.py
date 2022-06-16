import numpy as np

from pr3d.nonbayesian import ConditionalGaussianMM
from utils import create_dataset

gmm_model = ConditionalGaussianMM(
    h5_addr = "gmm_model.h5",
)
gmm_model.mlp.model.summary()

np.random.seed(0)
X,Y = create_dataset(n_samples = 10000, x_dim = 3)
print("X shape: {0}".format(X.shape))
print("Y shape: {0}".format(Y.shape))

print("Single test x: {0}, and y: {1}".format(X[10,:],Y[10]))
pdf,log_pdf,ecdf = gmm_model.prob_single(X[10,:],Y[10])
print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

print("Batch test x: {0}, and y: {1}".format(X[10:15,:],Y[10:15]))
pdf,log_pdf,ecdf = gmm_model.prob_batch(X[10:15,:],Y[10:15])
print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))