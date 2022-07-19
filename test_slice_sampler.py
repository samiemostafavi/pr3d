import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

import numpy as np

dtype = np.float32

mix = 0.3
target = tfd.Mixture(
    cat=tfd.Categorical(probs=[mix, 1.-mix]),
    components=[
        tfd.Normal(loc=-1., scale=0.1),
        tfd.Normal(loc=+1., scale=0.5),
    ],
)

samples = tfp.mcmc.sample_chain(
    num_results=1000,
    current_state=dtype(1),
    kernel=tfp.mcmc.SliceSampler(
        target.log_prob,
        step_size=1.0,
        max_doublings=5),
    num_burnin_steps=500,
    trace_fn=None,
    seed=1234)

sample_mean = tf.reduce_mean(samples, axis=0)
sample_std = tf.sqrt(
    tf.reduce_mean(
        tf.math.squared_difference(samples, sample_mean),
        axis=0))

print('Sample mean: ', sample_mean.numpy())
print('Sample Std: ', sample_std.numpy())