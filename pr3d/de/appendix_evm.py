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

def bounded_tanh(x, lo=-0.1, hi=2.0):
    return lo + (hi - lo) * (tf.math.tanh(x) + 1.) / 2.

tf.keras.utils.get_custom_objects()['bounded_tanh'] = bounded_tanh

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

        # create parameters dict
        self._params_config = {
            "tail_parameter": {
                "slice_size": 1,
                "slice_activation": "bounded_tanh", #"linear", #softplus
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
            p       = 0.99,
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
            # save bayesian
            hf.create_dataset("bayesian", shape=(1,), data=int(self.bayesian))

            # save bulk params
            for key, val in self._bulk_params.items():
                hf.create_dataset(key, data=np.array(val, dtype=np.float64))

            # save batch_size
            if self.batch_size is not None:
                hf.create_dataset("batch_size", shape=(1,), data=int(self.batch_size))

    def create_models(self):

        # now lets define the models to get probabilities

        # define dummy input
        self.dummy_input = self.core_model.input_layer
        # t = tf.fill(tf.shape(self.core_model.input_layer), 0.0)

        # put tensor components together (from X)
        self.tail_param = self.core_model.output_slices["tail_parameter"]
        #self.tail_threshold = self.core_model.output_slices["tail_threshold"]
        self.tail_scale = self.core_model.output_slices["tail_scale"]


        # create gaussian mixture prob model
        self.weights = tf.convert_to_tensor(np.array(self._bulk_params['mixture_weights']), dtype=self.dtype)
        self.locs = tf.convert_to_tensor(np.array(self._bulk_params['mixture_locations']), dtype=self.dtype)
        self.scales = tf.convert_to_tensor(np.array(self._bulk_params['mixture_scales']), dtype=self.dtype)
        cat = tfd.Categorical(probs=self.weights, dtype=self.dtype)
        components = [
            tfd.Normal(loc=loc, scale=scale)
            for loc, scale in zip(
                self.locs, self.scales
            )
        ]
        mixture = tfd.Mixture(cat=cat, components=components)

        # overwrite tail threshold
        #q99_const = tf.constant(self._q99, dtype=self.dtype)            # shape []
        #q99_broadcast = tf.broadcast_to(q99_const, tf.shape(self.tail_param))
        #self.tail_threshold = tf.stop_gradient(q99_broadcast)

        u_min_const   = tf.constant(self._q99, dtype=self.dtype)  # scalar
        u_min_bcast   = tf.broadcast_to(u_min_const,
                                        tf.shape(self.core_model.output_slices["tail_threshold"]))
        u_raw = self.core_model.output_slices["tail_threshold"]
        self.tail_threshold = u_min_bcast + tf.nn.softplus(u_raw)
        
        # these models are used for printing parameters
        self._params_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                self.tail_param,
                self.tail_threshold,
                self.tail_scale,
            ],
            name="params_model",
        )

        # find the normalization factor (from X)
        # squeezing the tail_threshold was important
        self.norm_factor = tf.constant(1.00, dtype=self.dtype) - mixture.cdf(
            tf.squeeze(self.tail_threshold)
        )
        self.norm_factor = tf.maximum(self.norm_factor,
                              tf.constant(1e-40, self.dtype))

        # define Y input
        self.y_input = keras.Input(
            name="y_input",
            shape=(1),
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

        # create batch size tensor (from Y)
        self.y_batchsize = tf.cast(tf.size(self.y_input), dtype=self.dtype)

        # split the values into bulk and tail according to the tail_threshold (from X and Y)
        bool_split_tensor, tail_samples_count, bulk_samples_count = split_bulk_gpd(
            tail_threshold=self.tail_threshold,
            y_input=self.y_input,
            y_batch_size=self.y_batchsize,
            dtype=self.dtype,
        )

        # define bulk probabilities (from X and Y)
        bulk_prob_t = mixture.prob(tf.squeeze(self.y_input))
        bulk_cdf_t = mixture.cdf(tf.squeeze(self.y_input))
        bulk_tail_prob_t = tf.constant(1.00, dtype=self.dtype) - bulk_cdf_t

        # define tail probabilities (from X and Y)
        gpd_prob_t = gpd_prob(
            tail_threshold=self.tail_threshold,
            tail_param=self.tail_param,
            tail_scale=self.tail_scale,
            norm_factor=self.norm_factor,
            y_input=tf.squeeze(self.y_input),
            dtype=self.dtype,
        )
        gpd_tail_prob_t = gpd_tail_prob(
            tail_threshold=self.tail_threshold,
            tail_param=self.tail_param,
            tail_scale=self.tail_scale,
            norm_factor=self.norm_factor,
            y_input=tf.squeeze(self.y_input),
            dtype=self.dtype,
        )

        # define final mixture probability tensors (from X and Y)
        self.pdf = mixture_prob(
            bool_split_tensor=bool_split_tensor,
            gpd_prob_t=gpd_prob_t,
            bulk_prob_t=bulk_prob_t,
            dtype=self.dtype,
        )

        self.log_pdf = mixture_log_prob(
            bool_split_tensor=bool_split_tensor,
            gpd_prob_t=gpd_prob_t,
            bulk_prob_t=bulk_prob_t,
            dtype=self.dtype,
        )
        self.expanded_log_pdf = tf.expand_dims(self.log_pdf, axis=1)

        self.ecdf = tf.constant(1.00, dtype=self.dtype) - mixture_tail_prob(
            bool_split_tensor=bool_split_tensor,
            gpd_tail_prob_t=gpd_tail_prob_t,
            bulk_tail_prob_t=bulk_tail_prob_t,
            dtype=self.dtype,
        )

        # these models are used for probability predictions
        self.full_prob_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.y_input,
            ],
            outputs=[
                tf.cast(bool_split_tensor, dtype=self.dtype),
                tf.cast(tf.logical_not(bool_split_tensor), dtype=self.dtype),
                bulk_prob_t,
                gpd_prob_t,
                tail_samples_count,
                bulk_samples_count,
            ],
            name="full_prob_model",
        )

        self._prob_pred_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.y_input,
            ],
            outputs=[self.pdf, self.log_pdf, self.ecdf],
            name="prob_pred_model",
        )

        self.norm_factor_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                tf.expand_dims(
                    self.norm_factor, axis=0
                ),  # very important "expand_dims"
            ],
            name="norm_factor_model",
        )

        # pipeline training model
        self._pl_training_model = keras.Model(
            inputs={"dummy_input": self.dummy_input, "y_input": self.y_input},
            outputs=[
                self.expanded_log_pdf,  # in shape: (batch_size,1)
            ],
        )


        # normal training model
        self._training_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.y_input,
            ],
            outputs=[
                self.expanded_log_pdf,  # in shape: (batch_size,1)
            ],
        )

        # defne the loss function
        # y_pred will be self.log_pdf which is (batch_size,1)
        self._loss = lambda y_true, y_pred: -tf.reduce_sum(y_pred)

        # create the sampling model
        # sample_input: random uniform numbers in [0,1]
        # feed them to the inverse cdf of the distribution

        # we use the threshold in CDF domain which is norm_factor and create cdf_bool_split_t
        # split sample_input into the ones greater or smaller than norm_factor
        # feed smallers to the icdf of Gamma, feed larger values to the icdf of GPD

        """
        # define random input
        self.sample_input = keras.Input(
                name = "sample_input",
                shape=(1),
                #batch_size = 100,
                dtype=self.dtype,
        )

        # split the samples into bulk and tail according to the norm_factor (from X and Y)
        cdf_bool_split_t = split_bulk_gpd_cdf(
            norm_factor = tf.constant(1.00,dtype=self.dtype)-self.norm_factor,
            random_input = self.sample_input,
            dtype = self.dtype,
        )

        # get gpd samples
        gpd_sample_t = gpd_quantile(
            tail_threshold = self.tail_threshold,
            tail_param = self.tail_param,
            tail_scale = self.tail_scale,
            norm_factor = self.norm_factor,
            random_input = tf.squeeze(self.sample_input),
            dtype = self.dtype,
        )

        # get bulk samples
        # ONLY WORKS WITH tf.compat.v1.disable_eager_execution()
        bulk_sample_t = gamma.quantile(
            tf.squeeze(self.sample_input)
        )

        # pass them through the mixture filter
        self.sample = mixture_sample(
            cdf_bool_split_t = cdf_bool_split_t,
            gpd_sample_t = gpd_sample_t,
            bulk_sample_t = bulk_sample_t,
            dtype = self.dtype,
        )

        self._sample_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.sample_input,
            ],
            outputs=[
                self.sample,
            ],
            name="sample_model",
        )
        """

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

