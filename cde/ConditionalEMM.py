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
        logits = y_pred[:,0:centers-1]
        locs = y_pred[:,centers:2*centers-1]
        scales = y_pred[:,2*centers:3*centers-1]
        tail_threshold = y_pred[:,3*centers]
        tail_param = y_pred[:,3*centers+1]
        tail_scale = y_pred[:,3*centers+2]

        y_batchsize = tf.cast(tf.size(y_true),dtype=dtype)

        # build mixture density
        cat = tfd.Categorical(logits=logits,dtype=dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(locs, axis=1), tf.unstack(scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # split the values into buld and tail according to the tail
        bool_split_tensor, tail_samples_count, bulk_samples_count = split_bulk_gpd(
            tail_threshold = tail_threshold,
            y_input = y_true,
            y_batch_size = y_batchsize,
            dtype = dtype,
        )

        # find the normalization factor
        norm_factor = tf.constant(1.00,dtype=dtype)-mixture.cdf(tail_threshold)

        # define bulk probabilities
        bulk_prob = mixture.prob(y_true)

        # define GPD log probability
        gpd_prob = gpd_prob(
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
            gpd_prob = gpd_prob,
            bulk_prob = bulk_prob,
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
        dtype : tf.DType = tf.float64,
    ):
        self.dtype = dtype

        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, 'r') as hf:
                self.centers = hf.get('centers')
                self.x_dim = hf.get('x_dim')
                self.hidden_sizes = hf.get('hidden_sizes')

            # first, read ConditionalGMM parameters from h5 file
            self.centers = 8

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

    def prob_single(self, x : npt.NDArray[np.float64], y : np.float64) -> Tuple[np.float64, np.float64, np.float64]:

        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        # y : np.float64 number
        [pdf,log_pdf,ecdf] = self.prob_pred_model([np.expand_dims(x, axis=0),np.expand_dims(y, axis=0)], training=False)
        return np.squeeze(pdf.numpy()),np.squeeze(log_pdf.numpy()),np.squeeze(ecdf.numpy())



    def prob_batch(self, x : npt.NDArray[np.float64], y : npt.NDArray[np.float64]):

        # for large batches of input x
        # x : np.array of np.float64 with the shape (?,ndim)
        # y : np.array of np.float64 with the shape (?)
        [pdf,log_pdf,ecdf] = self.prob_pred_model.predict(
            [x,y],
            batch_size=None,
            verbose=0,
            steps=None,
            callbacks=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
        )
        return pdf,log_pdf,ecdf


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
                    'mixture_locations': { 
                        'slice_size' : self.centers,
                        'slice_activation' : None,
                    },
                    'mixture_scales': { 
                        'slice_size' : self.centers,
                        'slice_activation' : 'softplus',
                    },
                    'mixture_weights': { 
                        'slice_size' : self.centers,
                        'slice_activation' : 'softmax',
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
                dtype=self.dtype,
            )  
        #self.mlp.model.summary()

        # now lets define the models to get probabilities

        # define Y input
        self.y_input = keras.Input(
                name = "y_input",
                shape=(1),
                dtype=self.dtype,
        )

        # create batch size tensor
        self.y_batchsize = tf.cast(tf.size(self.y_input),dtype=self.dtype)

        # define X input
        self.x_input = self.mlp.input_layer

        # put mixture components together
        self.logits = self.mlp.output_slices['mixture_weights']
        self.locs = self.mlp.output_slices['mixture_locations']
        self.scales = self.mlp.output_slices['mixture_scales']
        self.tail_threshold = self.mlp.output_slices['tail_parameter']
        self.tail_param = self.mlp.output_slices['tail_threshold']
        self.tail_scale = self.mlp.output_slices['tail_scale']

        cat = tfd.Categorical(logits=self.logits,dtype=self.dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)


        # split the values into buld and tail according to the tail
        bool_split_tensor, self.tail_samples_count, self.bulk_samples_count = split_bulk_gpd(
            tail_threshold = self.tail_threshold,
            y_input = self.y_input,
            y_batch_size = self.batch_size,
            dtype = self.dtype,
        )

        # find the normalization factor
        self.norm_factor = tf.constant(1.00,dtype=self.dtype)-mixture.cdf(self.tail_threshold)

        # define bulk probabilities
        bulk_prob = mixture.prob(self.y_input)
        bulk_cdf = mixture.cdf(self.y_input)
        bulk_tail_prob = tf.constant(1.00,dtype=self.dtype)-bulk_cdf

        # define GPD probabilities
        gpd_prob = gpd_prob(
            tail_threshold=self.tail_threshold,
            tail_param = self.tail_param,
            tail_scale = self.tail_scale,
            norm_factor = self.norm_factor,
            y_input = self.y_input,
            dtype = self.dtype,
        )
        gpd_tail_prob = gpd_tail_prob(
            tail_threshold=self.tail_threshold,
            tail_param = self.tail_param,
            tail_scale = self.tail_scale,
            norm_factor = self.norm_factor,
            y_input = self.y_input,
            dtype = self.dtype,
        )

        self.pdf = mixture_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_prob = gpd_prob,
            bulk_prob = bulk_prob,
        )

        self.log_pdf =  mixture_log_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_tail_prob = gpd_tail_prob,
            bulk_tail_prob = bulk_tail_prob,
        )

        self.ecdf = tf.constant(1.00,dtype=self.dtype) - mixture_tail_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_tail_prob = gpd_tail_prob,
            bulk_tail_prob = bulk_tail_prob,
        )

        # these models are used for probability predictions
        self.bulk_pred_model = keras.Model(
            inputs=self.x_input,
            outputs=[
                self.logits,
                self.locs,
                self.scales,
            ],
            name="bulk_pred_model",
        )
        self.gpd_pred_model = keras.Model(
            inputs=self.x_input,
            outputs=[
                self.tail_threshold,
                self.tail_param,
                self.tail_scale,
                self.norm_factor,
            ],
            name="gpd_pred_model",
        )
        self.mixture_model = keras.Model(
            inputs=[
                self.x_input,
                self.y_input
            ],
            outputs=[
                self.tail_samples_count, 
                self.bulk_samples_count,
            ],
            name="mixture_model",
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
        learning_rate : float = 5e-3,
        weight_decay : float = 0.0,
        epsilon : float = 1e-8,
    ):
        
        # define optimizer and train_step
        optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, epsilon=epsilon)

        # this keras model is the one that we use for training
        self.mlp.model.compile(optimizer=optimizer, loss=emm_nll_loss(self.centers,self.dtype))

        history = self.mlp.model.fit(
            X,
            Y,
            batch_size=1000,
            epochs=10,
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
