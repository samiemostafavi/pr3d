import h5py
import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_probability as tfp

from pr3d.common.core import ConditionalRecurrentDensityEstimator
from pr3d.common.evm import (
    gpd_prob,
    gpd_quantile,
    gpd_tail_prob,
    mixture_log_prob,
    mixture_prob,
    mixture_tail_prob,
    split_bulk_gpd,
)

tfd = tfp.distributions


class ConditionalRecurrentGaussianMixtureEVM(ConditionalRecurrentDensityEstimator):
    def __init__(
        self,
        centers: int = 3,
        x_dim: list = None,
        recurrent_taps: int = 32,
        hidden_layers_config : dict = {
            "hidden_lstm_1":{
                "type":"lstm",
                "size":64,
            },
            "hidden_lstm_2":{
                "type":"lstm",
                "size":64,
            },
            "hidden_dense_1":{
                "type":"dense",
                "size":16,
                "activation":"tanh",
            },
            "hidden_dense_2":{
                "type":"dense",
                "size":16,
                "activation":"tanh",
            }
        },
        h5_addr: str = None,
        batch_size: int = None,
        dtype: str = "float64",
    ):

        super(ConditionalRecurrentGaussianMixtureEVM, self).__init__(
            h5_addr=h5_addr,
            x_dim=x_dim,
            batch_size=batch_size,
            recurrent_taps=recurrent_taps,
            hidden_layers_config=hidden_layers_config,
            dtype=dtype,
        )

        # figure out parameters
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, "r") as hf:
                self._centers = int(hf.get("centers")[0])
                self._recurrent_taps = int(hf.get("recurrent_taps")[0])

                if "batch_size" in hf.keys():
                    self._batch_size = int(hf.get("batch_size")[0])

        else:
            self._centers = centers
            self._recurrent_taps = recurrent_taps
            self._batch_size = batch_size
            self._hidden_layers_config = hidden_layers_config
            self._x_dim = x_dim

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
            "tail_parameter": {
                "slice_size": 1,
                "slice_activation": "softplus",
            },
            "tail_threshold": {
                "slice_size": 1,
                "slice_activation": "softplus",
            },
            "tail_scale": {
                "slice_size": 1,
                "slice_activation": "softplus",
            },
        }

        # ask ConditionalDensityEstimator to form the MLP
        self.create_core(h5_addr=h5_addr)
        # self.core_model.model.summary()

        # create models for inference:
        # self._prob_pred_model, self._sample_model, self._params_model, self._training_model
        self.create_models()

    def save(self, h5_addr: str) -> None:
        self.core_model.model.save(h5_addr)
        with h5py.File(h5_addr, "a") as hf:
            hf.create_dataset("centers", shape=(1,), data=int(self.centers))
            hf.create_dataset("recurrent_taps", shape=(1,), data=int(self.recurrent_taps))
            if self.batch_size is not None:
                hf.create_dataset("batch_size", shape=(1,), data=int(self.batch_size))

    def create_models(self):

        # define input
        self.input = list(self.core_model.input_slices.values())

        # put mixture components together
        # put mixture components together
        self.weights = self.core_model.output_slices["mixture_weights"]
        self.locs = self.core_model.output_slices["mixture_locations"]
        self.scales = self.core_model.output_slices["mixture_scales"]
        self.tail_param = self.core_model.output_slices["tail_parameter"]
        self.tail_threshold = self.core_model.output_slices["tail_threshold"]
        self.tail_scale = self.core_model.output_slices["tail_scale"]

        # these models are used for printing parameters
        self._params_model = keras.Model(
            inputs=self.input,
            outputs=[
                self.weights,
                self.locs,
                self.scales,
                self.tail_param,
                self.tail_threshold,
                self.tail_scale,
            ],
            name="params_model",
        )

        # create gaussian mixture prob model
        cat = tfd.Categorical(probs=self.weights, dtype=self.dtype)
        components = [
            tfd.Normal(loc=loc, scale=scale)
            for loc, scale in zip(
                tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1)
            )
        ]
        mixture = tfd.Mixture(cat=cat, components=components)

        # find the normalization factor (from X)
        # squeezing the tail_threshold was important
        self.norm_factor = tf.constant(1.00, dtype=self.dtype) - mixture.cdf(
            tf.squeeze(self.tail_threshold)
        )

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

        # define GPD probabilities (from X and Y)
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
                *self.input,
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
                *self.input,
                self.y_input,
            ],
            outputs=[self.pdf, self.log_pdf, self.ecdf],
            name="prob_pred_model",
        )

        self.norm_factor_model = keras.Model(
            inputs=self.input,
            outputs=[
                tf.expand_dims(
                    self.norm_factor, axis=0
                ),  # very important "expand_dims"
            ],
            name="norm_factor_model",
        )

        # normal training model
        self._training_model = keras.Model(
            inputs=[
                *self.input,
                self.y_input,
            ],
            outputs=[
                self.expanded_log_pdf,  # in shape: (batch_size,1)
            ],
        )

        # defne the loss function
        # y_pred will be self.log_pdf which is (batch_size,1)
        class CustomLossLayer(tf.keras.layers.Layer):
            def __init__(self, idtype=tf.float64, **kwargs):
                super(CustomLossLayer, self).__init__(**kwargs)
                self.idtype = idtype

            def call(self, inputs):
                y_true, y_pred = inputs
                loss = -tf.reduce_sum(y_pred) / tf.cast(tf.size(y_true), self.idtype)
                return loss
        
        self._loss = lambda y_true, y_pred: CustomLossLayer(self.dtype)([y_true, y_pred])

    @property
    def centers(self):
        return self._centers

    def bulk_mean(
        self,
        x: npt.NDArray[np.float64],
    ):

        prediction_res = self._params_model.predict(
            x,
        )
        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        # put mixture components together
        mixture_weights_t = tf.convert_to_tensor(
            result_dict["mixture_weights"], dtype=self.dtype
        )
        mixture_locations_t = tf.convert_to_tensor(
            result_dict["mixture_locations"], dtype=self.dtype
        )
        mixture_scales_t = tf.convert_to_tensor(
            result_dict["mixture_scales"], dtype=self.dtype
        )

        # create gaussian mixture
        cat = tfd.Categorical(probs=mixture_weights_t, dtype=self.dtype)
        components = [
            tfd.Normal(loc=loc, scale=scale)
            for loc, scale in zip(
                tf.unstack(mixture_locations_t, axis=1),
                tf.unstack(mixture_scales_t, axis=1),
            )
        ]
        mixture = tfd.Mixture(cat=cat, components=components)

        return mixture.mean()

    def quantile(
        self,
        x,  # dict as below
        samples,  # numbers between 0.0 and 1.0 (numpy array)
        value_tolerance=1e-7,
        position_tolerance=1e-3,
    ):
        """
        vectorized numerical quantile finder
        """
        # x = { 'queue_length1': np.zeros(1000), 'queue_length2': np.zeros(1000), 'queue_length3' : np.zeros(1000) }
        x_list = np.array([np.array([*items]) for items in zip(*x.values())])

        def model_cdf_fn_t(x):
            a = x_list
            b = samples
            pdf, logpdf, cdf = self.prob_batch(x=a, y=x)
            return tf.convert_to_tensor(cdf - b, dtype=self.dtype)

        result = tfp.math.find_root_secant(
            objective_fn=model_cdf_fn_t,
            initial_position=self.bulk_mean(x),
            value_tolerance=tf.convert_to_tensor(
                np.ones(len(samples)) * value_tolerance, dtype=self.dtype
            ),
            position_tolerance=tf.convert_to_tensor(
                np.ones(len(samples)) * position_tolerance, dtype=self.dtype
            ),
        )

        return np.array(result[0])

    def sample_n(
        self,
        x,
        seed: int = 0,
    ):
        """
        https://stats.stackexchange.com/questions/243392/generate-sample-data-from-gaussian-mixture-model
        """
        # x = { 'queue_length1': np.zeros(1000), 'queue_length2': np.zeros(1000), 'queue_length3' : np.zeros(1000) }
        batch_size = len(list(x.values())[0])

        prediction_res = self._params_model.predict(
            x,
        )
        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        # print(result_dict)

        weights = result_dict["mixture_weights"]
        # weights_t = tf.convert_to_tensor(
        #     result_dict["mixture_weights"], dtype=self.dtype
        # )
        locs_t = tf.convert_to_tensor(
            result_dict["mixture_locations"], dtype=self.dtype
        )
        scales_t = tf.convert_to_tensor(result_dict["mixture_scales"], dtype=self.dtype)
        tail_param_t = tf.convert_to_tensor(
            result_dict["tail_parameter"], dtype=self.dtype
        )
        tail_threshold_t = tf.convert_to_tensor(
            result_dict["tail_threshold"], dtype=self.dtype
        )
        tail_scale_t = tf.convert_to_tensor(result_dict["tail_scale"], dtype=self.dtype)

        # select from random components
        cat_samples = tf.random.categorical(
            logits=tf.math.log(weights),
            num_samples=1,
            seed=seed,
        )
        cat_samples = tf.squeeze(cat_samples)
        locs_t = tf.gather(locs_t, cat_samples, axis=1, batch_dims=1)
        scales_t = tf.gather(scales_t, cat_samples, axis=1, batch_dims=1)
        components = tfd.Normal(loc=locs_t, scale=scales_t)
        tail_threshold_prob_t = components.cdf(tf.squeeze(tail_threshold_t))

        # random number in (0,1)
        y_samples = np.random.uniform(
            size=batch_size,
        )
        # generate bulk samples
        bulk_samples_t = components.quantile(y_samples)
        # generate tail samples
        gpd_samples_t = gpd_quantile(
            tail_threshold=tail_threshold_t,
            tail_param=tail_param_t,
            tail_scale=tail_scale_t,
            norm_factor=tf.constant(1.00, dtype=self.dtype) - tail_threshold_prob_t,
            random_input=tf.convert_to_tensor(y_samples),
            dtype=self.dtype,
        )

        # multiplex samples
        bool_split_t = tf.greater(y_samples, tail_threshold_prob_t)
        gpd_multiplexer = bool_split_t
        bulk_multiplexer = tf.logical_not(bool_split_t)
        gpd_multiplexer = tf.cast(
            gpd_multiplexer, dtype=self.dtype
        )  # convert it to float for multiplication
        bulk_multiplexer = tf.cast(
            bulk_multiplexer, dtype=self.dtype
        )  # convert it to float for multiplication
        multiplexed_gpd_samples = tf.multiply(gpd_samples_t, gpd_multiplexer)
        multiplexed_bulk_samples = tf.multiply(bulk_samples_t, bulk_multiplexer)

        return tf.reduce_sum(
            tf.stack(
                [
                    multiplexed_gpd_samples,
                    multiplexed_bulk_samples,
                ]
            ),
            axis=0,
        ).numpy()
