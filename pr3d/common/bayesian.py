import tensorflow_probability as tfp

# Bayesian Neural Network Layer Object
class SavableDenseFlipout(tfp.layers.DenseFlipout):
    def __init__(self,
        units,
        batch_size=None,
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
        if batch_size is None:
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)
        else:
            kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(batch_size * 1.0)

        # bias functions and divergence
        bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn()
        bias_posterior_tensor_fn=lambda d: d.sample()
        bias_prior_fn=tfp.layers.util.default_multivariate_normal_fn
        if batch_size is None:
            bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)
        else:
            bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p)/(batch_size * 1.0)

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