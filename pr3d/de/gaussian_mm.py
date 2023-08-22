import h5py
import keras
import tensorflow as tf
import tensorflow_probability as tfp

from pr3d.common.core import NonConditionalDensityEstimator

tfd = tfp.distributions


class GaussianMM(NonConditionalDensityEstimator):
    def __init__(
        self,
        centers: int = 8,
        h5_addr: str = None,
        bayesian: bool = False,
        batch_size: int = None,
        dtype: str = "float64",
    ):

        super(GaussianMM, self).__init__(
            h5_addr=h5_addr,
            bayesian=bayesian,
            batch_size=batch_size,
            dtype=dtype,
        )

        # figure out parameters
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, "r") as hf:
                self._centers = int(hf.get("centers")[0])
                self._bayesian = bool(hf.get("bayesian")[0])

                if "batch_size" in hf.keys():
                    self._batch_size = int(hf.get("batch_size")[0])

        else:
            self._centers = centers
            self._bayesian = bayesian
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
            hf.create_dataset("centers", shape=(1,), data=int(self.centers))
            hf.create_dataset("bayesian", shape=(1,), data=int(self.bayesian))
            if self.batch_size is not None:
                hf.create_dataset("batch_size", shape=(1,), data=int(self.batch_size))

    def create_models(self):

        # now lets define the models to get probabilities

        # define X input
        self.dummy_input = self.core_model.input_layer

        # put mixture components together
        self.weights = self.core_model.output_slices["mixture_weights"]
        self.locs = self.core_model.output_slices["mixture_locations"]
        self.scales = self.core_model.output_slices["mixture_scales"]

        # create params model
        self._params_model = keras.Model(
            inputs=self.dummy_input,
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

        # define Y input
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
                self.dummy_input,
                self.y_input,
            ],
            outputs=[self.pdf, self.log_pdf, self.ecdf],
            name="prob_pred_model",
        )

        # pipeline training model
        self._pl_training_model = keras.Model(
            inputs={"dummy_input": self.dummy_input, "y_input": self.y_input},
            outputs=[
                self.log_pdf,  # in shape: (batch_size,1)
            ],
        )

        # normal training model
        self._training_model = keras.Model(
            inputs=[
                self.dummy_input,
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
