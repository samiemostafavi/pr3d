import h5py
import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_probability as tfp

from pr3d.common.core import ConditionalRecurrentDensityEstimator

tfd = tfp.distributions



class ConditionalRecurrentGaussianMM(ConditionalRecurrentDensityEstimator):
    def __init__(
        self,
        centers: int = 8,
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

        super(ConditionalRecurrentGaussianMM, self).__init__(
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
                "slice_activation": "softmax"
            },
            "mixture_locations": {
                "slice_size": self.centers,
                "slice_activation": None
            },
            "mixture_scales": {
                "slice_size": self.centers,
                "slice_activation": "softplus"
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
            hf.create_dataset("centers", shape=(1,), data=int(self.centers))
            hf.create_dataset("recurrent_taps", shape=(1,), data=int(self.recurrent_taps))
            if self.batch_size is not None:
                hf.create_dataset("batch_size", shape=(1,), data=int(self.batch_size))

    def create_models(self):

        # now lets define the models to get probabilities

        # define sequence inputs: covariates and target sequence
        #self.input = self.core_model.input_layer
        self.input = list(self.core_model.input_slices.values())

        # put mixture components together
        self.weights = self.core_model.output_slices["mixture_weights"]
        self.locs = self.core_model.output_slices["mixture_locations"]
        self.scales = self.core_model.output_slices["mixture_scales"]

        # create params model
        self._params_model = keras.Model(
            inputs=self.input,
            outputs=[
                self.weights,
                self.locs,
                self.scales,
            ],
            name="params_model",
        )

        # create prob model
        cat = tfd.Categorical(probs=self.weights, dtype=self.dtype)
        components = [
            tfd.Normal(loc=loc, scale=scale)
            for loc, scale in zip(
                tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1)
            )
        ]
        mixture = tfd.Mixture(cat=cat, components=components)

        # define target value input y
        self.y_input = keras.Input(
            name="y_input",
            shape=(1),
            batch_size=self.batch_size,
            dtype=self.dtype,
        )

        # define pdf, logpdf and loglikelihood
        # CAUTION: tfd.Mixture needs this transpose, otherwise, would give (100,100)
        self.pdf = tf.transpose(mixture.prob(tf.transpose(self.y_input)))
        self.log_pdf = tf.transpose(mixture.log_prob(tf.transpose(self.y_input)))
        self.ecdf = tf.transpose(mixture.cdf(tf.transpose(self.y_input)))

        # these models are used for probability predictions
        self._prob_pred_model = keras.Model(
            inputs=[
                *self.input,
                self.y_input,
            ],
            outputs=[self.pdf, self.log_pdf, self.ecdf],
            name="prob_pred_model",
        )

        # normal training model
        self._training_model = keras.Model(
            inputs=[
                *self.input,
                self.y_input,
            ],
            outputs=[
                self.log_pdf,
            ],
        )

        class CustomLossLayer(tf.keras.layers.Layer):
            def __init__(self, idtype=tf.float64, **kwargs):
                super(CustomLossLayer, self).__init__(**kwargs)
                self.idtype = idtype

            def call(self, inputs):
                y_true, y_pred = inputs
                loss = -tf.reduce_sum(y_pred) / tf.cast(tf.size(y_true), self.idtype)
                return loss

        # define the loss function
        # y_pred will be self.log_pdf which is (batch_size,1)
        #self._loss = lambda y_true, y_pred: -tf.reduce_sum(y_pred)/tf.cast(tf.size(self.y_input),self.dtype)
        self._loss = lambda y_true, y_pred: CustomLossLayer(self.dtype)([y_true, y_pred])


    @property
    def centers(self):
        return self._centers
    

    def sample_n_parallel(
        self,
        Y,
        X,
        n: int,
        seed: int = 0,
    ) -> npt.NDArray[np.float64]:
        
        # generate n random numbers uniformly distributed on [0,1]
        # Y : np array with length equal to self.recurrent_taps
        if len(Y) != self.recurrent_taps or len(X) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        prediction_res = self._params_model.predict(
            [np.expand_dims(Y, axis=0), np.expand_dims(X, axis=0)]
        )
        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        weights = result_dict["mixture_weights"]
        locs_t = tf.convert_to_tensor(
            result_dict["mixture_locations"], dtype=self.dtype
        )
        scales_t = tf.convert_to_tensor(result_dict["mixture_scales"], dtype=self.dtype)

        # select from components
        cat_samples = tf.random.categorical(
            logits=tf.expand_dims(tf.math.log(weights),axis=0),
            num_samples=1,
            seed=seed,
        )

        locs_t = tf.gather(tf.expand_dims(locs_t,axis=0), cat_samples, axis=1, batch_dims=1)
        scales_t = tf.gather(tf.expand_dims(scales_t,axis=0), cat_samples, axis=1, batch_dims=1)

        components = tfd.Normal(loc=locs_t, scale=scales_t)

        # random numbers in (0,1)
        y_samples = np.random.uniform(
            size=n,
        )

        result = components.quantile(y_samples)
        return np.squeeze(result.numpy())

    def quantile(
        self,
        Y,
        X,
        q,
        seed: int = 0,
    ) -> npt.NDArray[np.float64]:
        
        # Y : np array with length equal to self.recurrent_taps
        if len(Y) != self.recurrent_taps or len(X) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        prediction_res = self._params_model.predict(
            [np.expand_dims(Y, axis=0), np.expand_dims(X, axis=0)]
        )
        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        weights = result_dict["mixture_weights"]
        locs_t = tf.convert_to_tensor(
            result_dict["mixture_locations"], dtype=self.dtype
        )
        scales_t = tf.convert_to_tensor(result_dict["mixture_scales"], dtype=self.dtype)

        # select from components
        cat_samples = tf.random.categorical(
            logits=tf.expand_dims(tf.math.log(weights),axis=0),
            num_samples=1,
            seed=seed,
        )

        locs_t = tf.gather(tf.expand_dims(locs_t,axis=0), cat_samples, axis=1, batch_dims=1)
        scales_t = tf.gather(tf.expand_dims(scales_t,axis=0), cat_samples, axis=1, batch_dims=1)

        components = tfd.Normal(loc=locs_t, scale=scales_t)

        result = components.quantile(np.array([q]))
        return np.squeeze(result.numpy())
    

    def quantile_batch(
        self,
        Y,
        X,
        q,
        seed: int = 0,
    ) -> npt.NDArray[np.float64]:
        
        # Y : np array with length equal to self.recurrent_taps
        if len(Y[0,:]) != self.recurrent_taps or len(X[0,:]) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        prediction_res = self._params_model.predict(
            [Y, X]
        )
        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        weights = result_dict["mixture_weights"]
        locs_t = tf.convert_to_tensor(
            result_dict["mixture_locations"], dtype=self.dtype
        )
        scales_t = tf.convert_to_tensor(result_dict["mixture_scales"], dtype=self.dtype)

        # select from components
        cat_samples = tf.random.categorical(
            logits=tf.math.log(weights),
            num_samples=1,
            seed=seed,
        )

        locs_t = tf.gather(tf.squeeze(locs_t), tf.squeeze(cat_samples), axis=1, batch_dims=1)
        scales_t = tf.gather(tf.squeeze(scales_t), tf.squeeze(cat_samples), axis=1, batch_dims=1)

        components = tfd.Normal(loc=locs_t, scale=scales_t)

        result = components.quantile(q)
        return np.squeeze(result.numpy())
