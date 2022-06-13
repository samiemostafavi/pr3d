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

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def dense_var_prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def dense_var_posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


class SavableDenseVariational(tfp.layers.DenseVariational):
    def __init__(self,
               units,
               make_posterior_fn,
               make_prior_fn,
               kl_weight=None,
               kl_use_exact=False,
               activation=None,
               use_bias=True,
               activity_regularizer=None,
               **kwargs) -> None:

        super(SavableDenseVariational,self).__init__(
            units,
            make_posterior_fn,
            make_prior_fn,
            kl_weight,
            kl_use_exact,
            activation,
            use_bias,
            activity_regularizer,
            **kwargs)

        self._kl_weight = kl_weight
        self._kl_use_exact = kl_use_exact

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'make_posterior_fn': self._make_posterior_fn,
            'make_prior_fn': self._make_prior_fn,
            'kl_weight': self._kl_weight,
            'kl_use_exact': self._kl_use_exact,
            'activation': self.activation,
            'use_bias' : self.use_bias,
            'activity_regularizer' : self.activity_regularizer,

        })
        return config


def create_probablistic_bnn_model(
    train_size, 
    feature_names, 
    hidden_units,
    dtype=tf.float32,
):
    inputs = create_model_inputs(feature_names)
    features = keras.layers.concatenate(list(inputs.values()))
    features = layers.BatchNormalization()(features)

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        #features = tfp.layers.DenseVariational(
        features = SavableDenseVariational(
            units=units,
            make_prior_fn=dense_var_prior,
            make_posterior_fn=dense_var_posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
            dtype=dtype,
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    distribution_params = layers.Dense(
        units=2,
        dtype=dtype,
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
            'SavableDenseVariational' : SavableDenseVariational,
            'dense_var_prior' : dense_var_prior,
            'dense_var_posterior' : dense_var_posterior,
            'negative_loglikelihood' : negative_loglikelihood,
            'loss' : negative_loglikelihood
        }
    )
    return model
