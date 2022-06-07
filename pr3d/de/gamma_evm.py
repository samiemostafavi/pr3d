import tensorflow as tf
import keras
import numpy as np
import numpy.typing as npt
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from typing import Tuple
tfd = tfp.distributions

from .core import SLP
from .ev_mixture_tf import *

# in order to use tfd.Gamma.quantile
tf.compat.v1.disable_eager_execution()

class GammaEVM():
    def __init__(
        self,
        h5_addr : str = None,
        dtype : str = 'float64',
    ):
        # configure keras to use dtype
        tf.keras.backend.set_floatx(dtype)

        # for creating the tensors
        if dtype == 'float64':
            self.dtype = tf.float64
        elif dtype == 'float32':
            self.dtype = tf.float32
        elif dtype == 'float16':
            self.dtype = tf.float16
        else:
            raise Exception("unknown dtype format")

        if h5_addr is not None:

            # load the keras model
            loaded_slp_model = keras.models.load_model(h5_addr, custom_objects={ 'loss' : gevm_nll_loss(self.dtype) })
            #self._model.summary()

            # create the model
            self.create_model(loaded_slp_model)
        else:

            # create the slp model
            self.create_model()

    def get_parameters(self) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64 , np.float64]:

        # for single value x (not batch)
        # y : np.float64 number
        x = 0
        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        [gamma_shape,gamma_rate] = self.bulk_param_model.predict(np.expand_dims(x, axis=0))
        [tail_threshold,tail_param,tail_scale] = self.gpd_param_model.predict(np.expand_dims(x, axis=0))
        norm_factor = self.norm_factor_model.predict(np.expand_dims(x, axis=0))

        return {
            "gamma_shape" : np.squeeze(gamma_shape),
            "gamma_rate" : np.squeeze(gamma_rate),
            "tail_threshold" : np.squeeze(tail_threshold),
            "tail_param" : np.squeeze(tail_param),
            "tail_scale" : np.squeeze(tail_scale),
            "norm_factor": np.squeeze(norm_factor),
        }

    def prob_single(self, y : np.float64) -> Tuple[np.float64, np.float64, np.float64]:

        # for single value x (not batch)
        # y : np.float64 number
        x = 0
        [pdf,log_pdf,ecdf] = self.prob_pred_model.predict(
            [np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)],
            #batch_size=None,
            #verbose=0,
            #steps=None,
            #callbacks=None,
            #max_queue_size=10,
            #workers=1,
            #use_multiprocessing=False,
        )
        return np.squeeze(pdf),np.squeeze(log_pdf),np.squeeze(ecdf)


    def prob_batch(self, y : npt.NDArray[np.float64]):
        
        # for large batches of input y
        # y : np.array of np.float64 with the shape (batch_size,1) e.g. np.array([5,6,7,8,9,10])
        x = np.zeros(len(y))
        [pdf,log_pdf,ecdf] = self.prob_pred_model.predict(
            [x,y],
            #batch_size=None,
            #verbose=0,
            #steps=None,
            #callbacks=None,
            #max_queue_size=10,
            #workers=1,
            #use_multiprocessing=False,
        )
        return np.squeeze(pdf),np.squeeze(log_pdf),np.squeeze(ecdf)

    def sample_n(self, 
        n : int, 
        random_generator: np.random.Generator = np.random.default_rng(),
    ) -> npt.NDArray[np.float64]:

        # generate n random numbers uniformly distributed on [0,1]
        x = np.zeros(n)
        y = random_generator.uniform(0,1,n)

        samples = self.sample_model.predict(
            [x,y],
            #batch_size=None,
            #verbose=0,
            #steps=None,
            #callbacks=None,
            #max_queue_size=10,
            #workers=1,
            #use_multiprocessing=False,
        )
        return np.squeeze(samples)

    def create_model(self, loaded_slp_model : keras.Model = None):

        if loaded_slp_model is not None:
            self.slp = SLP(
                loaded_slp_model = loaded_slp_model
            )
        else:
            self.slp = SLP(
                name = 'evm_keras_model',
                layer_config={
                    'gamma_shape': { 
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    },
                    'gamma_rate': { 
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    },
                    'tail_parameter' : {
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    },
                    'tail_threshold' : {
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    },
                    'tail_scale' : {
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    }
                },
                #batch_size = 100,
                dtype = self.dtype
            )  
        #self.slp.model.summary()

        # now lets define the models to get probabilities

        # define dummy input
        self.dummy_input = self.slp.input_layer

        # put mixture components together (from X)
        self.gamma_shape = self.slp.output_slices['gamma_shape']
        self.gamma_rate = self.slp.output_slices['gamma_rate']
        self.tail_param = self.slp.output_slices['tail_parameter']
        self.tail_threshold = self.slp.output_slices['tail_threshold']
        self.tail_scale = self.slp.output_slices['tail_scale']

        # build gamma density (from X)
        gamma = tfd.Gamma(concentration=tf.squeeze(self.gamma_shape), rate=tf.squeeze(self.gamma_rate))

        # find the normalization factor (from X)
        # squeezing the tail_threshold was important
        self.norm_factor = tf.constant(1.00,dtype=self.dtype)-gamma.cdf(tf.squeeze(self.tail_threshold))

        # define Y input
        self.y_input = keras.Input(
                name = "y_input",
                shape=(1),
                #batch_size = 100,
                dtype=self.dtype,
        )

        # create batch size tensor (from Y)
        self.y_batchsize = tf.cast(tf.size(self.y_input),dtype=self.dtype)

        # split the values into bulk and tail according to the tail_threshold (from X and Y)
        bool_split_tensor, tail_samples_count, bulk_samples_count = split_bulk_gpd(
            tail_threshold = self.tail_threshold,
            y_input = self.y_input,
            y_batch_size = self.y_batchsize,
            dtype = self.dtype,
        )

        # define bulk probabilities (from X and Y)
        bulk_prob_t = gamma.prob(tf.squeeze(self.y_input))
        bulk_cdf_t = gamma.cdf(tf.squeeze(self.y_input))
        bulk_tail_prob_t = tf.constant(1.00,dtype=self.dtype)-bulk_cdf_t


        # define tail probabilities (from X and Y)
        gpd_prob_t = gpd_prob(
            tail_threshold=self.tail_threshold,
            tail_param = self.tail_param,
            tail_scale = self.tail_scale,
            norm_factor = self.norm_factor,
            y_input = tf.squeeze(self.y_input),
            dtype = self.dtype,
        )
        gpd_tail_prob_t = gpd_tail_prob(
            tail_threshold=self.tail_threshold,
            tail_param = self.tail_param,
            tail_scale = self.tail_scale,
            norm_factor = self.norm_factor,
            y_input = tf.squeeze(self.y_input),
            dtype = self.dtype,
        )

        # define final mixture probability tensors (from X and Y)
        self.pdf = mixture_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_prob_t = gpd_prob_t,
            bulk_prob_t = bulk_prob_t,
            dtype = self.dtype,
        )

        self.log_pdf =  mixture_log_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_prob_t = gpd_prob_t,
            bulk_prob_t = bulk_prob_t,
            dtype = self.dtype,
        )
        self.ecdf = tf.constant(1.00,dtype=self.dtype) - mixture_tail_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_tail_prob_t = gpd_tail_prob_t,
            bulk_tail_prob_t = bulk_tail_prob_t,
            dtype = self.dtype,
        )

        # these models are used for probability predictions
        self.full_prob_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.y_input,
            ],
            outputs=[
                tf.cast(bool_split_tensor,dtype=self.dtype),
                tf.cast(tf.logical_not(bool_split_tensor),dtype=self.dtype),
                bulk_prob_t, 
                gpd_prob_t,
                tail_samples_count,
                bulk_samples_count,
            ],
            name="full_prob_model",
        )

        self.prob_pred_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.y_input,
            ],
            outputs=[
                self.pdf,
                self.log_pdf,
                self.ecdf
            ],
            name="prob_pred_model",
        )

        # these models are used for printing paramters
        self.bulk_param_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                self.gamma_shape,
                self.gamma_rate,
            ],
            name="bulk_param_model",
        )
        self.gpd_param_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                self.tail_threshold,
                self.tail_param,
                self.tail_scale,
            ],
            name="gpd_param_model",
        )
        self.norm_factor_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                tf.expand_dims(self.norm_factor, axis=0), # very important "expand_dims"
            ],
            name="norm_factor_model",
        )

        # create the sampling model
        # sample_input: random uniform numbers in [0,1]
        # feed them to the inverse cdf of the distribution

        # we use the threshold in CDF domain which is norm_factor and create cdf_bool_split_t
        # split sample_input into the ones greater or smaller than norm_factor
        # feed smallers to the icdf of Gamma, feed larger values to the icdf of GPD

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

        self.sample_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.sample_input,
            ],
            outputs=[
                self.sample,
            ],
            name="sample_model",
        )

    
     
    def fit(self, 
        Y,
        batch_size : int = 1000,
        epochs : int = 10,
        learning_rate : float = 5e-3,
        weight_decay : float = 0.0,
        epsilon : float = 1e-8,
    ):

        learning_rate = np.cast[self.dtype.as_numpy_dtype](learning_rate)
        weight_decay = np.cast[self.dtype.as_numpy_dtype](weight_decay)
        epsilon = np.cast[self.dtype.as_numpy_dtype](epsilon)

        # define optimizer and train_step
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=epsilon)

        # this keras model is the one that we use for training
        self.slp.model.compile(optimizer=optimizer, loss=gevm_nll_loss(self.dtype))

        X = np.zeros(len(Y))
        history = self.slp.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            #validation_data=(x_val, y_val),
        )

    def save(self, h5_addr : str) -> None:
        self.slp.model.save(h5_addr)
