import h5py
import keras
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import numpy.typing as npt

from pr3d.common.core import NonConditionalDensityEstimator
from pr3d.common.evm import (
    gpd_prob,
    gpd_tail_prob,
    gpd_quantile,
    mixture_log_prob,
    mixture_prob,
    mixture_tail_prob,
    split_bulk_gpd,
)

tfd = tfp.distributions

from keras.saving import register_keras_serializable

@register_keras_serializable(package="custom")
def bounded_tanh(x, lo=-0.5, hi=0.6):
    return lo + (hi - lo) * (tf.math.tanh(x) + 1.) / 2.

def make_bounded_tanh(lo, hi):
    def fn(x, lo=lo, hi=hi):
        lo = tf.cast(lo, x.dtype); hi = tf.cast(hi, x.dtype)
        return lo + (hi - lo) * (tf.math.tanh(x) + 1.0) / 2.0
    return fn

# in order to use tfd.Gamma.quantile
# tf.compat.v1.disable_eager_execution()

from scipy.stats import norm
from scipy.optimize import bisect

def gaussian_mixture_quantile(weights: np.ndarray,
                              locs:    np.ndarray,
                              scales:  np.ndarray,
                              p: float = 0.99,
                              bracket_sigmas: float = 10.0) -> float:
    """
    Returns q such that F(q)=p for a 1-D Gaussian mixture
    defined by weights, locs, scales  (all 1-D arrays of equal length).
    """

    # normalise weights in case they do not sum to 1 exactly
    weights = weights / weights.sum()

    def mix_cdf(x: float) -> float:
        return np.sum(weights * norm.cdf((x - locs) / scales))

    # crude but safe bracket
    lo = locs.min() - bracket_sigmas * scales.max()
    hi = locs.max() + bracket_sigmas * scales.max()

    # root-find F(x) - p = 0
    return bisect(lambda x: mix_cdf(x) - p, lo, hi, xtol=1e-8)

#def bounded_tanh(x, lo=-0.05, hi=2.0):
#    return lo + (hi - lo) * (tf.math.tanh(x) + 1.) / 2.

#tf.keras.utils.get_custom_objects()['bounded_tanh'] = bounded_tanh

class AppendixEVM(NonConditionalDensityEstimator):
    def __init__(
        self,
        bulk_params: dict = None,
        h5_addr: str = None,
        tanh_lo: float = -0.5,
        tanh_hi: float = 0.6,
        param_threshold: float = 0.99,
        bayesian: bool = False,
        batch_size: int = None,
        dtype: str = "float64",
    ):
        super(AppendixEVM, self).__init__(
            h5_addr=h5_addr,
            bayesian=bayesian,
            batch_size=batch_size,
            dtype=dtype,
        )

        # figure out parameters
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, "r") as hf:

                # load bayesian
                self._bayesian = bool(hf.get("bayesian")[0])
                self._tanh_lo = bool(hf.get("tanh_lo")[0])
                self._tanh_hi = bool(hf.get("tanh_hi")[0])
                self._param_threshold = bool(hf.get("param_threshold")[0])

                # load bulk_params
                self._bulk_params = {}
                for ds in hf.keys():
                    if 'mixture_weights' in ds:
                        self._bulk_params = {
                            **self._bulk_params,
                            'mixture_weights' : hf.get('mixture_weights')[:]
                        }
                    if 'mixture_locations' in ds:
                        self._bulk_params = {
                            **self._bulk_params,
                            'mixture_locations' : hf.get('mixture_locations')[:]
                        }
                    if 'mixture_scales' in ds:
                        self._bulk_params = {
                            **self._bulk_params,
                            'mixture_scales' : hf.get('mixture_scales')[:]
                        }

                # load batch_size
                if "batch_size" in hf.keys():
                    self._batch_size = int(hf.get("batch_size")[0])
        else:
            self._bulk_params = bulk_params
            self._bayesian = bayesian
            self._batch_size = batch_size
            self._tanh_lo = tanh_lo
            self._tanh_hi = tanh_hi
            self._param_threshold = param_threshold

        # create parameters dict
        self._params_config = {
            "tail_parameter": {
                "slice_size": 1,
                "slice_activation": make_bounded_tanh(self._tanh_lo,self._tanh_hi), #"linear", #softplus
                "slice_kernel_initializer": "zeros",
                "slice_bias_initializer":   "zeros",
            },
            "tail_threshold": {
                "slice_size": 1,
                "slice_activation": None, #"softplus",
                "slice_kernel_initializer": "zeros",
                "slice_bias_initializer":   "zeros",
            },
            "tail_scale": {
                "slice_size": 1,
                "slice_activation": "softplus",
            },
        }

        self._q99 = gaussian_mixture_quantile(
            weights = np.asarray(self._bulk_params['mixture_weights'],   dtype=np.float64),
            locs    = np.asarray(self._bulk_params['mixture_locations'], dtype=np.float64),
            scales  = np.asarray(self._bulk_params['mixture_scales'],    dtype=np.float64),
            p       = self._param_threshold,
        )

        # ask NonConditionalDensityEstimator to form the SLP
        self.create_core(h5_addr=h5_addr)
        # self._core_model.model.summary()

        # create models for inference:
        # self._prob_pred_model, self._sample_model, self._params_model, self._training_model
        self.create_models()

    def save(self, h5_addr: str) -> None:
        self.core_model.model.save(h5_addr)
        with h5py.File(h5_addr, "a") as hf:
            hf.create_dataset("tanh_lo", shape=(1,), data=int(self._tanh_lo))
            hf.create_dataset("tanh_hi", shape=(1,), data=int(self._tanh_hi))
            hf.create_dataset("param_threshold", shape=(1,), data=int(self._param_threshold))

            # save bayesian
            hf.create_dataset("bayesian", shape=(1,), data=int(self.bayesian))

            # save bulk params
            for key, val in self._bulk_params.items():
                hf.create_dataset(key, data=np.array(val, dtype=np.float64))

            # save batch_size
            if self.batch_size is not None:
                hf.create_dataset("batch_size", shape=(1,), data=int(self.batch_size))

    def create_models(self):

        # --- inputs ---
        self.dummy_input = self.core_model.input_layer  # keep your SLP input
        self.y_input = keras.Input(
            name="y_input",
            shape=(1,),                      # FIX: (1,) not (1)
            dtype=self.dtype,
        )

        # --- bulk (constant) Gaussian mixture from provided params ---
        self.weights = tf.convert_to_tensor(np.array(self._bulk_params['mixture_weights']), dtype=self.dtype)
        self.locs    = tf.convert_to_tensor(np.array(self._bulk_params['mixture_locations']), dtype=self.dtype)
        self.scales  = tf.convert_to_tensor(np.array(self._bulk_params['mixture_scales']), dtype=self.dtype)

        # tfd.MixtureSameFamily is simpler & vectorized
        mix = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self.weights),  # don't pass float dtype
            components_distribution=tfd.Normal(loc=self.locs, scale=self.scales),
        )

        # --- tail parameter heads from SLP ---
        self.tail_param  = self.core_model.output_slices["tail_parameter"]   # (None,1)
        u_raw            = self.core_model.output_slices["tail_threshold"]   # (None,1)
        self.tail_scale  = self.core_model.output_slices["tail_scale"]       # (None,1)

        # --- effective threshold (already fixed unique name) ---
        self.tail_threshold = layers.Lambda(
            lambda x: tf.nn.softplus(x) + tf.cast(self._q99, x.dtype),
            name="tail_threshold_eff",
            output_shape=(1,),                      # per-sample one value (still (None,1) here)
        )(u_raw)

        # --- params model ---
        self._params_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[self.tail_param, self.tail_threshold, self.tail_scale],
            name="params_model",
        )

        # ---- norm factor: 1 - F_bulk(u) -> (None,) then clamp ----
        norm_raw = layers.Lambda(
            lambda u: 1.0 - mix.cdf(tf.squeeze(u, axis=-1)),
            name="norm_factor_raw",
            output_shape=(None,),
        )(self.tail_threshold)
        self.norm_factor = layers.Lambda(
            lambda n: tf.maximum(n, tf.cast(1e-40, n.dtype)),
            name="norm_factor",
            output_shape=(None,),
        )(norm_raw)

        # ---- y flatten: (None,1) -> (None,) ----
        y_flat = layers.Lambda(lambda y: tf.squeeze(y, axis=-1),
                              name="y_flat", output_shape=(None,))(self.y_input)

        # ---- bulk prob / cdf (ALL -> (None,)) ----
        bulk_prob_t = layers.Lambda(lambda y: mix.prob(y),
                                    name="bulk_prob", output_shape=(None,))(y_flat)
        bulk_cdf_t  = layers.Lambda(lambda y: mix.cdf(y),
                                    name="bulk_cdf",  output_shape=(None,))(y_flat)
        bulk_tail_prob_t = layers.Lambda(lambda c: 1.0 - c,
                                        name="bulk_tail_prob", output_shape=(None,))(bulk_cdf_t)

        # ---- GPD prob / tail (return (None,)) ----
        gpd_prob_t = layers.Lambda(
            lambda args: gpd_prob(args[0], args[1], args[2], args[3], args[4], dtype=self.dtype),
            name="gpd_prob", output_shape=(None,),
        )([self.tail_threshold, self.tail_param, self.tail_scale, self.norm_factor, y_flat])

        gpd_tail_prob_t = layers.Lambda(
            lambda args: gpd_tail_prob(args[0], args[1], args[2], args[3], args[4], dtype=self.dtype),
            name="gpd_tail_prob", output_shape=(None,),
        )([self.tail_threshold, self.tail_param, self.tail_scale, self.norm_factor, y_flat])

        # ---- split: (None,) boolean ----
        bool_split_tensor = layers.Lambda(
            lambda args: tf.greater(tf.squeeze(args[0], -1), tf.squeeze(args[1], -1)),
            name="is_tail", output_shape=(None,),
        )([self.y_input, self.tail_threshold])

        # (Optional) counts (scalars, but keep as (None,) just to avoid extra rank issues)
        tail_samples_count = layers.Lambda(
            lambda b: tf.reduce_sum(tf.cast(b, self.dtype)),
            name="tail_count", output_shape=(),
        )(bool_split_tensor)
        batch_size_t = layers.Lambda(
            lambda y: tf.cast(tf.shape(y)[0], self.dtype),
            name="batch_size_t", output_shape=(),
        )(self.y_input)
        bulk_samples_count = layers.Lambda(
            lambda ab: ab[0] - ab[1],
            name="bulk_count", output_shape=(),
        )([batch_size_t, tail_samples_count])

        # ---- final mixture PDF / tail  (all inputs (None,) → output (None,)) ----
        self.pdf = layers.Lambda(
            lambda args: tf.where(args[0], args[1], tf.zeros_like(args[1])) +
                        tf.where(tf.logical_not(args[0]), args[2], tf.zeros_like(args[2])),
            name="mixture_pdf", output_shape=(None,),
        )([bool_split_tensor, gpd_prob_t, bulk_prob_t])

        self.log_pdf = layers.Lambda(
            lambda z: tf.math.log(tf.maximum(z, tf.constant(1e-40, z.dtype))),
            name="mixture_logpdf", output_shape=(None,),
        )(self.pdf)

        # training wants (None,1)
        self.expanded_log_pdf = layers.Lambda(
            lambda z: tf.expand_dims(z, -1),
            name="expanded_log_pdf", output_shape=(None,1),
        )(self.log_pdf)

        mixture_tail = layers.Lambda(
            lambda args: tf.where(args[0], args[1], tf.zeros_like(args[1])) +
                        tf.where(tf.logical_not(args[0]), args[2], tf.zeros_like(args[2])),
            name="mixture_tail", output_shape=(None,),
        )([bool_split_tensor, gpd_tail_prob_t, bulk_tail_prob_t])

        self.ecdf = layers.Lambda(lambda t: 1.0 - t,
                                  name="ecdf_scalar", output_shape=(None,))(mixture_tail)
        self.ecdf = layers.Lambda(lambda z: tf.expand_dims(z, -1),
                                  name="ecdf", output_shape=(None,1))(self.ecdf)

        # turn the boolean mask into float via Lambda layers (shape => (None,))
        is_tail_float = layers.Lambda(
            lambda b: tf.cast(b, self.dtype),
            name="is_tail_float",
            output_shape=(None,),
        )(bool_split_tensor)

        is_bulk_float = layers.Lambda(
            lambda b: tf.cast(tf.logical_not(b), self.dtype),
            name="is_bulk_float",
            output_shape=(None,),
        )(bool_split_tensor)

        self.full_prob_model = keras.Model(
            inputs=[self.dummy_input, self.y_input],
            outputs=[
                is_tail_float,
                is_bulk_float,
                bulk_prob_t,       # (None,)
                gpd_prob_t,        # (None,)
                tail_samples_count,  # scalar ()
                bulk_samples_count,  # scalar ()
            ],
            name="full_prob_model",
        )

        self._prob_pred_model = keras.Model(
            inputs=[self.dummy_input, self.y_input],
            outputs=[self.pdf, self.log_pdf, self.ecdf],
            name="prob_pred_model",
        )

        self.norm_factor_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[layers.Lambda(lambda n: tf.expand_dims(n, 0),
                                  name="norm_factor_expand", output_shape=(None,))(self.norm_factor)],
            name="norm_factor_model",
        )

        self._pl_training_model = keras.Model(
            inputs=[self.dummy_input, self.y_input],
            outputs=[self.expanded_log_pdf],
            name="pl_training_model",
        )

        self._training_model = keras.Model(
            inputs=[self.dummy_input, self.y_input],
            outputs=[self.expanded_log_pdf],
            name="training_model",
        )

        self._loss = lambda y_true, y_pred: -tf.reduce_mean(y_pred)


    @property
    def centers(self):
        return self._centers

    def bulk_mean(
        self
    ):

        # create gaussian mixture prob model
        self.weights = tf.convert_to_tensor(np.array(self._bulk_params['mixture_weights'], dtype=self.dtype), dtype=self.dtype)
        self.locs = tf.convert_to_tensor(np.array(self._bulk_params['mixture_locations'], dtype=self.dtype), dtype=self.dtype)
        self.scales = tf.convert_to_tensor(np.array(self._bulk_params['mixture_scales'], dtype=self.dtype), dtype=self.dtype)
        cat = tfd.Categorical(probs=self.weights, dtype=self.dtype)
        components = [
            tfd.Normal(loc=loc, scale=scale)
            for loc, scale in zip(
                tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1)
            )
        ]
        mixture = tfd.Mixture(cat=cat, components=components)

        return mixture.mean()