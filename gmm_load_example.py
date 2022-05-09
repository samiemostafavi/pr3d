import keras
from cde import ConditionalGMM
from cde import create_dataset


gmm_model = ConditionalGMM(
    centers = 8,
    x_dim = 3,
    hidden_sizes = (16,16),
    learning_rate = 1e-2,
    weight_decay = 0.0,
    epsilon = 1e-8,
)

gmm_model.mlp._model = keras.models.load_model("gmm_model.h5", custom_objects={ 'loss': gmm_model.negative_log_likelihood_loss })

X,Y = create_dataset(n_samples = 10000, x_dim = 3)
print("X shape: {0}".format(X.shape))
print("Y shape: {0}".format(Y.shape))

print("Single test x: {0}, and y: {1}".format(X[10,:],Y[10]))
pdf,log_pdf,ecdf = gmm_model.prob_single(X[10,:],Y[10])
print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

print("Batch test x: {0}, and y: {1}".format(X[10:15,:],Y[10:15]))
pdf,log_pdf,ecdf = gmm_model.prob_batch(X[10:15,:],Y[10:15])
print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))