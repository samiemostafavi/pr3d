import h5py
import keras
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from typing import Tuple

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
from keras import layers

from keras.saving import register_keras_serializable

from pr3d.common.core import NonConditionalDensityEstimator

tfd = tfp.distributions
class DensityEstimator:
    def __init__(
        self,
        h5_addr: str = None,
        bayesian: bool = False,
        batch_size: int = None,
        dtype: str = "float64",
    ):
        self._bayesian = bayesian
        self._batch_size = batch_size

        # configure keras to use dtype
        tf.keras.backend.set_floatx(dtype)

        # for creating the tensors
        if dtype == "float64":
            self._dtype = tf.float64
        elif dtype == "float32":
            self._dtype = tf.float32
        elif dtype == "float16":
            self._dtype = tf.float16
        else:
            raise Exception("unknown dtype format")

    def create_core(self, h5_addr: str):
        pass

    def save(self, h5_addr: str):
        pass

    def create_models(self):
        pass

    def prob_single(self):
        pass

    def prob_batch(self):
        pass

    def sample_n(self):
        pass

    def get_parameters(self):
        pass

    def fit(self):
        pass

    @property
    def prob_pred_model(self) -> keras.Model:
        return self._prob_pred_model

    @property
    def sample_model(self) -> keras.Model:
        return self._sample_model

    @property
    def params_model(self) -> keras.Model:
        return self._params_model

    @property
    def training_model(self) -> keras.Model:
        return self._training_model

    @property
    def pl_training_model(self) -> keras.Model:
        return self._pl_training_model

    @property
    def params_config(self) -> dict:
        return self._params_config

    @property
    def core_model(self):
        return self._core_model

    @property
    def loss(self):
        return self._loss

    @property
    def bayesian(self):
        return self._bayesian

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dtype(self):
        return self._dtype




class GaussianMM(NonConditionalDensityEstimator):
    def __init__(
        self,
        centers: int = 8,
        h5_addr: str = None,
        batch_size: int = None,
        dtype: str = "float64",
    ):

        super(GaussianMM, self).__init__(
            h5_addr=h5_addr,
            batch_size=batch_size,
            dtype=dtype,
        )

        # figure out parameters
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, "r") as hf:
                self._centers = int(hf.get("centers")[0])

                if "y_mean" in hf and "y_std" in hf:
                    self._y_mean = float(hf["y_mean"][0])
                    self._y_std  = float(hf["y_std"][0])

                if "batch_size" in hf.keys():
                    self._batch_size = int(hf.get("batch_size")[0])

        else:
            self._centers = centers
            self._batch_size = batch_size

        # create parameters dict
        self._params_config = {
            "mixture_weights": {
                "slice_size": self.centers,
                "slice_activation": "softmax",
            },
            "mixture_locations": {
                "slice_size": self.centers,
                "slice_activation": None,
            },
            "mixture_scales": {
                "slice_size": self.centers,
                "slice_activation": "softplus",
            },
        }

        # ask NonConditionalDensityEstimator to form the SLP
        self.create_core(h5_addr=h5_addr)
        # self._core_model.model.summary()

        # create models for inference:
        # self._prob_pred_model, self._sample_model, self._params_model, self._training_model
        self.create_models()

    def save(self, h5_addr: str) -> None:
        self.core_model.model.save(h5_addr)
        with h5py.File(h5_addr, "a") as hf:
            hf.create_dataset("y_mean", data=np.array([y_mean], dtype=np.float64))
            hf.create_dataset("y_std",  data=np.array([y_std],  dtype=np.float64))
            hf.create_dataset("centers", shape=(1,), data=int(self.centers))
            if self.batch_size is not None:
                hf.create_dataset("batch_size", shape=(1,), data=int(self.batch_size))

    def create_models(self):
        # --- inputs & params ---
        self.dummy_input = self.core_model.input_layer  # shape (None,1)
        self.weights = self.core_model.output_slices["mixture_weights"]  # (None, C)
        self.locs    = self.core_model.output_slices["mixture_locations"]  # (None, C)
        self.scales  = self.core_model.output_slices["mixture_scales"]     # (None, C)

        # params model (unchanged)
        self._params_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[self.weights, self.locs, self.scales],
            name="params_model",
        )

        # y input: use tuple shape
        self.y_input = keras.Input(
            name="y_input",
            shape=(1,),                    # <-- FIX: (1,)
            dtype=self.dtype,
        )
        # flatten to scalar per example for TFP
        y_flat = layers.Lambda(lambda y: tf.squeeze(y, axis=-1), name="y_flat")(self.y_input)

        # Small helper building the distribution inside a Keras Lambda
        def _mk_mix(args):
            w, m, s = args
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=w),  # dtype defaults to int32 internally
                components_distribution=tfd.Normal(loc=m, scale=s),
            )
            return mix

        # We compute pdf/logpdf/cdf via Lambdas so Keras accepts it
        def _pdf(args):
            w, m, s, y = args
            mix = _mk_mix([w, m, s])
            return mix.prob(y)  # (None,)

        def _logpdf(args):
            w, m, s, y = args
            mix = _mk_mix([w, m, s])
            return mix.log_prob(y)  # (None,)

        def _cdf(args):
            w, m, s, y = args
            mix = _mk_mix([w, m, s])
            return mix.cdf(y)  # (None,)

        pdf     = layers.Lambda(_pdf,    name="pdf")    ([self.weights, self.locs, self.scales, y_flat])
        log_pdf = layers.Lambda(_logpdf, name="log_pdf")([self.weights, self.locs, self.scales, y_flat])
        ecdf    = layers.Lambda(_cdf,    name="ecdf")   ([self.weights, self.locs, self.scales, y_flat])

        # expand dims back to (None,1) like you had
        self.pdf     = layers.Lambda(lambda z: tf.expand_dims(z, -1))(pdf)
        self.log_pdf = layers.Lambda(lambda z: tf.expand_dims(z, -1))(log_pdf)
        self.ecdf    = layers.Lambda(lambda z: tf.expand_dims(z, -1))(ecdf)

        # models you use downstream
        self._prob_pred_model = keras.Model(
            inputs=[self.dummy_input, self.y_input],
            outputs=[self.pdf, self.log_pdf, self.ecdf],
            name="prob_pred_model",
        )

        self._pl_training_model = keras.Model(
            inputs={"dummy_input": self.dummy_input, "y_input": self.y_input},
            outputs=[self.log_pdf],
            name="pl_training_model",
        )

        self._training_model = keras.Model(
            inputs=[self.dummy_input, self.y_input],
            outputs=[self.log_pdf],
            name="training_model",
        )

        # loss: negative mean log-likelihood (keeps your signature)
        class CustomLossLayer(tf.keras.layers.Layer):
            def __init__(self, idtype=tf.float64, **kwargs):
                super().__init__(**kwargs)
                self.idtype = idtype
            def call(self, inputs):
                y_true, y_pred = inputs
                # y_pred is (batch,1); take mean over batch
                return -tf.reduce_mean(y_pred)

        self._loss = lambda y_true, y_pred: CustomLossLayer(self.dtype)([y_true, y_pred])
 
    @property
    def centers(self):
        return self._centers