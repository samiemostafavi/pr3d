import tensorflow as tf
import keras
import tensorflow_probability as tfp
import h5py
tfd = tfp.distributions

from pr3d.common.tf_core import SLP, NonConditionalDensityEstimator
from pr3d.common.gmm_core import *

class GaussianMM(NonConditionalDensityEstimator):
    def __init__(
        self,
        h5_addr : str = None,
        centers : int = 8,
        dtype : str = 'float64',
    ):

        super(GaussianMM,self).__init__(
            h5_addr = h5_addr,
            dtype = dtype,
        )

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

        # figure out centers
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, 'r') as hf:
                self._centers = hf.get('centers')
        else:
            self._centers = centers

        self._loss = gmm_nll_loss(self.centers,dtype)

        self._params_config = {
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
        }
            
        # initiate the slp model
        if h5_addr is not None:
            # load the keras model and feed to SLP
            self._slp_model = SLP(
                loaded_slp_model = keras.models.load_model(
                    h5_addr, 
                    custom_objects={ 
                        'loss' : self._loss
                    },
                )
            )

        else:
            # create SLP model
            self._slp_model = SLP(
                name = 'gmm_keras_model',
                layer_config=self._params_config,
                #batch_size = 100,
                dtype = self.dtype
            )

        #self._slp_model.model.summary()

        # create models for inference: 
        # self._prob_pred_model, self._sample_model, self._params_model
        self.create_models()

    @property
    def centers(self):
        return self._centers

    def save(self, h5_addr : str) -> None:
        self.slp_model.model.save(h5_addr)
        with h5py.File(h5_addr, 'a') as hf:
            hf.create_dataset('centers',self.centers)

    def create_models(self):

        # now lets define the models to get probabilities

        # define X input
        self.dummy_input = self.slp_model.input_layer

        # put mixture components together
        self.weights = self.slp_model.output_slices['mixture_weights']
        self.locs = self.slp_model.output_slices['mixture_locations']
        self.scales = self.slp_model.output_slices['mixture_scales']

        cat = tfd.Categorical(probs=self.weights,dtype=self.dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # define Y input
        self.y_input = keras.Input(
                name = "y_input",
                shape=(1),
                #batch_size = 100,
                dtype=self.dtype,
        )

        # define pdf, logpdf and loglikelihood
        self.pdf = mixture.prob(tf.squeeze(self.y_input))
        self.log_pdf = mixture.log_prob(tf.squeeze(self.y_input))
        self.ecdf = mixture.cdf(tf.squeeze(self.y_input))

        # these models are used for probability predictions
        self._prob_pred_model = keras.Model(
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
        self._params_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                self.weights,
                self.locs,
                self.scales,
            ],
            name="params_model",
        )

    
