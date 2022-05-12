import numpy as np

from cde import ConditionalEMM
from cde import create_dataset

dtype = 'float64', # 'float32' or 'float16'

emm_model = ConditionalEMM(
    centers = 2,
    x_dim = 3,
    hidden_sizes = (16,16),
    dtype = dtype,
)

np.random.seed(0)
X,Y = create_dataset(n_samples = 10000, x_dim = 3, dtype = dtype)
print("X shape: {0}".format(X.shape))
print("Y shape: {0}".format(Y.shape))


# train the model
emm_model.fit(
    X,Y,
    batch_size = 10000, # 1000
    epochs = 1000, # 10
    learning_rate = 5e-2,
    weight_decay = 0.0,
    epsilon = 1e-8,
)

weights,locs,scales,tail_threshold,tail_param,tail_scale,norm_factor = emm_model.verbose_param_batch(X[10:15,:])
print("weights: {0},locs: {1},scales: {2},tail_threshold: {3},tail_param: {4},tail_scale: {5},norm_factor: {6}".format(weights,locs,scales,tail_threshold,tail_param,tail_scale,norm_factor))


#print("Single test x: {0}, and y: {1}".format(X[10,:],Y[10]))
#pdf,log_pdf,ecdf = emm_model.prob_single(X[10,:],Y[10])
#print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

#print("Batch test x: {0}, and y: {1}".format(X[10:15,:],Y[10:15]))
#pdf,log_pdf,ecdf = emm_model.prob_batch(X[10:15,:],Y[10:15])
#print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

#weights,locs,scales,tail_threshold,tail_param,tail_scale,norm_factor = emm_model.verbose_param_batch(X[10:15,:])
#print("weights: {0},locs: {1},scales: {2},tail_threshold: {3},tail_param: {4},tail_scale: {5},norm_factor: {6}".format(weights,locs,scales,tail_threshold,tail_param,tail_scale,norm_factor))

(gpd_multiplexer,
    bulk_multiplexer,
    bulk_prob_t,
    gpd_prob_t,
    tail_samples_count,
    bulk_samples_count) = emm_model.verbose_prob_batch(X[10:15,:],Y[10:15])
print("gpd_multiplexer: {0}, bulk_multiplexer: {1}, bulk_prob_t: {2}, gpd_prob_t: {3}, tail_samples_count: {4}, bulk_samples_count: {5}".format(
        gpd_multiplexer,
        bulk_multiplexer,
        bulk_prob_t,
        gpd_prob_t,
        tail_samples_count,
        bulk_samples_count
    )
)

emm_model.save("emm_model.h5")
