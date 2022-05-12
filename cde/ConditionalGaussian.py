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


def gmm_nll_loss(centers,dtype):
    # Gaussian mixture network negative log likelihood loss
    def loss(y_true, y_pred):
        # y_pred is the concatenated mixture_weights, mixture_locations, and mixture_scales
        weights = y_pred[:,0:centers-1]
        locs = y_pred[:,centers:2*centers-1]
        scales = y_pred[:,2*centers:3*centers-1]

        # very important line, was causing (batch_size,batch_size)
        y_true = tf.squeeze(y_true)
        
        cat = tfd.Categorical(probs=weights,dtype=dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(locs, axis=1), tf.unstack(scales, axis=1))]

        mixture = tfd.Mixture(cat=cat, components=components)

        # define logpdf and loglikelihood
        log_pdf_ = mixture.log_prob(y_true)
        log_likelihood_ = tf.reduce_sum(log_pdf_ )
        return -log_likelihood_

    return loss

class ConditionalGMM():    
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
            loaded_mlp_model = keras.models.load_model(h5_addr, custom_objects={ 'loss' : gmm_nll_loss(self.centers) })
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
                name = 'gmm_keras_model',
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

        # define X input
        self.x_input = self.mlp.input_layer

        # put mixture components together
        self.weights = self.mlp.output_slices['mixture_weights']
        self.locs = self.mlp.output_slices['mixture_locations']
        self.scales = self.mlp.output_slices['mixture_scales']

        cat = tfd.Categorical(probs=self.weights,dtype=self.dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # define pdf, logpdf and loglikelihood
        self.pdf = mixture.prob(self.y_input)
        self.log_pdf = mixture.log_prob(self.y_input)
        self.ecdf = mixture.cdf(self.y_input)

        # these models are used for probability predictions
        self.theta_pred_model = keras.Model(inputs=self.x_input,outputs=[self.weights,self.locs,self.scales],name="theta_pred_model")
        self.prob_pred_model = keras.Model(inputs=[self.x_input,self.y_input],outputs=[self.pdf,self.log_pdf,self.ecdf],name="prob_pred_model")
     
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
        self.mlp.model.compile(optimizer=optimizer, loss=gmm_nll_loss(self.centers,self.dtype))

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
