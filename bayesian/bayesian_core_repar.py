import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp

def get_divergence_fn(
    batch_size,
):
    def divergence_fn(q,p,_):
        return tfp.distributions.kl_divergence(q, p) / (batch_size * 1.0)

    return divergence_fn

def create_model_inputs(
    feature_names,
    dtype=tf.float32
):
    inputs = {}
    for feature_name in feature_names:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), 
            dtype=dtype,
        )
    return inputs

def create_probablistic_bnn_model(
    batch_size,
    feature_names, 
    hidden_units,
    dtype=tf.float32,
):

    inputs = create_model_inputs(feature_names)
    features = keras.layers.concatenate(list(inputs.values()))
    #features = layers.BatchNormalization()(features)

    divergence_fn = get_divergence_fn(batch_size)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseReparameterization(
            units = units,
            kernel_divergence_fn=divergence_fn,
            bias_divergence_fn=divergence_fn,
            bias_prior_fn=tfp.layers.util.default_multivariate_normal_fn,
            bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.

    distribution_params = layers.Dense(
        units=2,
        dtype=dtype,
    )(features)
    #distribution_params = features
    outputs = tfp.layers.IndependentNormal(
        event_shape=1,
        dtype=dtype,
    )(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

def load_bnn_model(h5_addr : str, batch_size):
    divergence_fn = get_divergence_fn(batch_size)
    model = keras.models.load_model(
        h5_addr,
        custom_objects={ 
            'DenseReparameterization' : tfp.layers.DenseReparameterization,
            'divergence_fn' : divergence_fn,
            'negative_loglikelihood' : negative_loglikelihood,
            'loss' : negative_loglikelihood
        }
    )
    return model
