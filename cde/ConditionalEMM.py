import tensorflow as tf
import keras
import numpy as np
import numpy.typing as npt
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import h5py
from typing import Tuple
tfd = tfp.distributions

from .CoreNetwork import MLP
from .ev_mixture_tf import *



def emm_nll_loss(centers,dtype):
    # Mixture network + GPD (Extreme mixture model) negative log likelihood loss
    def loss(y_true, y_pred):

        # y_pred is the concatenated mixture_weights, mixture_locations, and mixture_scales
        weights = y_pred[:,0:centers-1]
        locs = y_pred[:,centers:2*centers-1]
        scales = y_pred[:,2*centers:3*centers-1]
        tail_param = y_pred[:,3*centers]
        tail_threshold = y_pred[:,3*centers+1]
        tail_scale = y_pred[:,3*centers+2]

        # very important line, was causing (batch_size,batch_size)
        y_true = tf.squeeze(y_true)

        # find y_batchsize
        y_batchsize = tf.cast(tf.size(y_true),dtype=dtype)

        # build mixture density
        cat = tfd.Categorical(probs=weights,dtype=dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(locs, axis=1), tf.unstack(scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # split the values into bulk and tail according to the threshold
        bool_split_tensor, tail_samples_count, bulk_samples_count = split_bulk_gpd(
            tail_threshold = tail_threshold,
            y_input = y_true,
            y_batch_size = y_batchsize,
            dtype = dtype,
        )

        # find the normalization factor
        #norm_factor = tf.constant(1.00,dtype=dtype)-mixture.cdf(tf.squeeze(tail_threshold))
        norm_factor = tf.constant(1.00,dtype=dtype)-mixture.cdf(tf.squeeze(tail_threshold))

        # define bulk probabilities
        bulk_prob_t = mixture.prob(y_true)

        # define GPD log probability
        gpd_prob_t = gpd_prob(
            tail_threshold=tail_threshold,
            tail_param = tail_param,
            tail_scale = tail_scale,
            norm_factor = norm_factor,
            y_input = y_true,
            dtype = dtype,
        )

        # define logpdf and loglikelihood
        log_pdf_ = mixture_log_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_prob_t = gpd_prob_t,
            bulk_prob_t = bulk_prob_t,
            dtype = dtype,
        )

        log_likelihood_ = tf.reduce_sum(log_pdf_)
        return -log_likelihood_

    return loss


class ConditionalEMM():
    def __init__(
        self,
        h5_addr : str = None,
        centers : int = 8,
        x_dim : int = 3,
        hidden_sizes : tuple = (16,16),
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
            # read side parameters
            with h5py.File(h5_addr, 'r') as hf:
                self.centers = hf.get('centers')
                self.x_dim = hf.get('x_dim')
                self.hidden_sizes = hf.get('hidden_sizes')

            # load the keras model
            loaded_mlp_model = keras.models.load_model(h5_addr, custom_objects={ 'loss' : emm_nll_loss(self.centers,self.dtype) })
            #self._model.summary()

            # create the model
            self.create_model(loaded_mlp_model)
        else:

            self.centers = centers
            self.x_dim = x_dim
            self.hidden_sizes = hidden_sizes

            # create the mlp model
            self.create_model()

    def verbose_param_batch(self, x : npt.NDArray[np.float64]):
        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        [weights,locs,scales] = self.bulk_param_model.predict(x)
        [tail_threshold,tail_param,tail_scale,norm_factor] = self.gpd_param_model.predict(x)

        return (
            np.squeeze(weights),
            np.squeeze(locs),
            np.squeeze(scales),
            np.squeeze(tail_threshold),
            np.squeeze(tail_param),
            np.squeeze(tail_scale),
            np.squeeze(norm_factor),
        )

    def verbose_prob_batch(self, x : npt.NDArray[np.float64], y : np.float64):
        [gpd_multiplexer,
            bulk_multiplexer,
            bulk_prob_t, 
            gpd_prob_t,
            tail_samples_count,
            bulk_samples_count] = self.full_prob_model.predict([x,y])

        return (
            gpd_multiplexer,
            bulk_multiplexer,
            bulk_prob_t, 
            gpd_prob_t,
            tail_samples_count,
            bulk_samples_count,
        )

    def prob_single(self, x : npt.NDArray[np.float64], y : np.float64) -> Tuple[np.float64, np.float64, np.float64]:

        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        # y : np.float64 number
        [pdf,log_pdf,ecdf] = self.prob_pred_model([np.expand_dims(x, axis=0),np.expand_dims(y, axis=0)], training=False)
        return np.squeeze(pdf.numpy()),np.squeeze(log_pdf.numpy()),np.squeeze(ecdf.numpy())



    def prob_batch(self, x : npt.NDArray[np.float64] , y : npt.NDArray[np.float64]):
        
        # for large batches of input x
        # x : np.array of np.float64 with the shape (batch_size,ndim)
        # y : np.array of np.float64 with the shape (batch_size,1)

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


    def create_model(self, loaded_mlp_model : keras.Model = None):

        if loaded_mlp_model is not None:
            self.mlp = MLP(
                loaded_mlp_model = loaded_mlp_model
            )
        else:
            self.mlp = MLP(
                name = 'emm_keras_model',
                input_shape=(self.x_dim),
                output_layer_config={
                    'mixture_weights': { 
                        'slice_size' : self.centers,
                        'slice_activation' : 'softmax',
                    },
                    'mixture_locations': { 
                        'slice_size' : self.centers,
                        'slice_activation' : None,
                    },
                    'mixture_scales': { 
                        'slice_size' : self.centers,
                        'slice_activation' : 'softplus',
                    },
                    'tail_parameter' : {
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    },
                    'tail_threshold' : {
                        'slice_size' : 1,
                        'slice_activation' : None,
                    },
                    'tail_scale' : {
                        'slice_size' : 1,
                        'slice_activation' : 'softplus',
                    }
                },
                hidden_sizes=self.hidden_sizes,
                hidden_activation='tanh',
                #batch_size= 100,
                dtype=self.dtype,
            )  
        #self.mlp.model.summary()


        # now lets define the models to get probabilities

        # define X input
        self.x_input = self.mlp.input_layer

        # put mixture components together (from X)
        self.weights = self.mlp.output_slices['mixture_weights']
        self.locs = self.mlp.output_slices['mixture_locations']
        self.scales = self.mlp.output_slices['mixture_scales']
        self.tail_param = self.mlp.output_slices['tail_parameter']
        self.tail_threshold = self.mlp.output_slices['tail_threshold']
        self.tail_scale = self.mlp.output_slices['tail_scale']

        # form the bulk density function (from X)
        cat = tfd.Categorical(probs=self.weights,dtype=self.dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # find the normalization factor (from X)
        # squeezing the tail_threshold was important
        self.norm_factor = tf.constant(1.00,dtype=self.dtype)-mixture.cdf(tf.squeeze(self.tail_threshold))

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
        bulk_prob_t = mixture.prob(tf.squeeze(self.y_input))
        bulk_cdf_t = mixture.cdf(tf.squeeze(self.y_input))
        bulk_tail_prob_t = tf.constant(1.00,dtype=self.dtype)-bulk_cdf_t


        # define GPD probabilities (from X and Y)
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
        self.bulk_param_model = keras.Model(
            inputs=self.x_input,
            outputs=[
                self.weights,
                self.locs,
                self.scales,
            ],
            name="bulk_param_model",
        )
        self.gpd_param_model = keras.Model(
            inputs=self.x_input,
            outputs=[
                self.tail_threshold,
                self.tail_param,
                self.tail_scale,
                self.norm_factor,
            ],
            name="gpd_param_model",
        )
        self.full_prob_model = keras.Model(
            inputs=[
                self.x_input,
                self.y_input
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
                self.x_input,
                self.y_input
            ],
            outputs=[
                self.pdf,
                self.log_pdf,
                self.ecdf
            ],
            name="prob_pred_model",
        )

    
     
    def fit(self, 
        X, Y,
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
        self.mlp.model.compile(optimizer=optimizer, loss=emm_nll_loss(self.centers,self.dtype))

        history = self.mlp.model.fit(
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
        self.mlp.model.save(h5_addr)
        with h5py.File(h5_addr, 'a') as hf:
            hf.create_dataset('centers',self.centers)
            hf.create_dataset('x_dim',self.x_dim)
            hf.create_dataset('hidden_sizes',self.hidden_sizes)
