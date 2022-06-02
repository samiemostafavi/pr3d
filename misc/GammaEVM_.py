
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.distributions import distribution
from tensorflow_probability.python.internal import parameter_properties
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import samplers

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


__all__ = [
    'gammaEVM',
]

class GammaEVM(distribution.AutoCompositeTensorDistribution):
    def __init__( self,
        gamma_shape,
        gamma_rate,
        tail_param,
        tail_threshold,
        tail_scale,
        dtype,
        validate_args=False,
        allow_nan_stats=True,
        parameters=None,
        name="gammaEVM"
    ):

        parameters = dict(locals())
        with tf.name_scope(name) as name:
            self._gamma_shape = tensor_util.convert_nonref_to_tensor(
                gamma_shape, dtype=dtype, name='gamma_shape')
            self._gamma_rate = tensor_util.convert_nonref_to_tensor(
                gamma_rate, dtype=dtype, name='gamma_rate')
            self._tail_param = tensor_util.convert_nonref_to_tensor(
                tail_param, dtype=dtype, name='tail_param')
            self._tail_threshold = tensor_util.convert_nonref_to_tensor(
                tail_threshold, dtype=dtype, name='tail_threshold')
            self._tail_scale = tensor_util.convert_nonref_to_tensor(
                tail_scale, dtype=dtype, name='tail_scale')

        super().__init__(
            dtype=dtype,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
            parameters=parameters,
            name = name,
        )


    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        # pylint: disable=g-long-lambda
        return dict(
            gamma_shape=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            gamma_rate=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            tail_param=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            tail_threshold=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
            tail_scale=parameter_properties.ParameterProperties(
                default_constraining_bijector_fn=(
                    lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        )
        # pylint: enable=g-long-lambda

    @property
    def gamma_shape(self):
        """Gamma shape parameter."""
        return self._gamma_shape 

    @property
    def gamma_rate(self):
        """Gamma rate parameter."""
        return self._gamma_rate

    @property
    def tail_param(self):
        """Gamma tail parameter."""
        return self._tail_param
    
    @property
    def tail_threshold(self):
        """Tail threshold parameter."""
        return self._tail_threshold

    @property
    def tail_scale(self):
        """Tail scale parameter."""
        return self._tail_scale

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _gamma_shape_parameter(self):
        return tf.convert_to_tensor(self.gamma_shape)

    def _gamma_rate_parameter(self):
        return tf.convert_to_tensor(self.gamma_rate)

    def _tail_param_parameter(self):
        return tf.convert_to_tensor(self.tail_param)

    def _tail_threshold_parameter(self):
        return tf.convert_to_tensor(self.tail_threshold)

    def _tail_scale_parameter(self):
        return tf.convert_to_tensor(self.tail_scale)

    def _sample_n(self, n, seed=None):
        gamma_shape = tf.convert_to_tensor(self.gamma_shape)
        gamma_rate = tf.convert_to_tensor(self.gamma_rate)
        tail_param = tf.convert_to_tensor(self.tail_param)
        tail_threshold = tf.convert_to_tensor(self.tail_threshold)
        tail_scale = tf.convert_to_tensor(self.tail_scale)

        shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)],
                        axis=0)
        sampled = samplers.normal(
            shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)
        return sampled * scale + loc