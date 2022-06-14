import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tensorflow_io as tfio


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


class SavableDenseFlipout(tfp.layers.DenseFlipout):
    def __init__(self,
        units,
        batch_size,
        activation=None,
        activity_regularizer=None,
        trainable=True,
        kernel_posterior_fn=None,
        kernel_posterior_tensor_fn=None,
        kernel_prior_fn=None,
        kernel_divergence_fn=None,
        bias_posterior_fn=None,
        bias_posterior_tensor_fn=None,
        bias_prior_fn=None,
        bias_divergence_fn=None,
        seed=None,
        **kwargs):

        # kernel functions and divergence
        kernel_posterior_fn=tfp.layers.util.default_mean_field_normal_fn()
        kernel_posterior_tensor_fn=lambda d: d.sample()
        kernel_prior_fn=tfp.layers.util.default_multivariate_normal_fn
        kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (batch_size * 1.0)

        # bias functions and divergence
        bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn()
        bias_posterior_tensor_fn=lambda d: d.sample()
        bias_prior_fn=tfp.layers.util.default_multivariate_normal_fn
        bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (batch_size * 1.0)

        super(SavableDenseFlipout,self).__init__(
            units=units,
            activation=activation,
            activity_regularizer=activity_regularizer,
            trainable=trainable,
            kernel_posterior_fn=kernel_posterior_fn,
            kernel_posterior_tensor_fn=kernel_posterior_tensor_fn,
            kernel_prior_fn=kernel_prior_fn,
            kernel_divergence_fn=kernel_divergence_fn,
            bias_posterior_fn=bias_posterior_fn,
            bias_posterior_tensor_fn=bias_posterior_tensor_fn,
            bias_prior_fn=bias_prior_fn,
            bias_divergence_fn=bias_divergence_fn,
            seed=seed,
            **kwargs)

        self._batch_size = batch_size

    def get_config(self):
        config = super(SavableDenseFlipout, self).get_config().copy()
        config.update({
            'batch_size': self._batch_size,
        })
        return config


def create_probablistic_bnn_model(
    batch_size,
    feature_names, 
    hidden_units,
    dtype=tf.float32,
):

    inputs = create_model_inputs(feature_names)
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)


    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = SavableDenseFlipout(
            units=units,
            batch_size=batch_size,
            activation="relu",
        )(features)

    # Create a probabilistic output (Normal distribution), and use another DenseFlipout layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = SavableDenseFlipout(
            units=2,
            batch_size=batch_size,
            activation="relu",
        )(features)
    
    outputs = tfp.layers.IndependentNormal(
        event_shape=1,
        dtype=dtype,
    )(distribution_params)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

def load_bnn_model(h5_addr : str):
    model = keras.models.load_model(
        h5_addr,
        custom_objects={ 
            'SavableDenseFlipout' : SavableDenseFlipout,
            'negative_loglikelihood' : negative_loglikelihood,
            'loss' : negative_loglikelihood
        }
    )
    return model
