import tensorflow as tf
import keras
import numpy as np
import numpy.typing as npt
from scipy import optimize
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import h5py
from typing import Tuple
tfd = tfp.distributions

from pr3d.common.core import ConditionalDensityEstimator

class ConditionalGaussianMM(ConditionalDensityEstimator):
    def __init__(
        self,
        centers : int = 8,
        x_dim : list = None,
        h5_addr : str = None,
        bayesian : bool = False,
        batch_size : int = None,
        dtype : str = 'float64',
        hidden_sizes = (16,16), 
        hidden_activation = 'tanh',
    ):

        super(ConditionalGaussianMM,self).__init__(
            x_dim = x_dim,
            h5_addr = h5_addr,
            bayesian = bayesian,
            batch_size = batch_size,
            dtype = dtype,
            hidden_sizes = hidden_sizes,
            hidden_activation = hidden_activation,
        )

        # figure out parameters
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, 'r') as hf:
                self._x_dim = [ encoded.decode("utf-8")  for encoded in list(hf.get('x_dim')[0])]
                self._centers = int(hf.get('centers')[0])
                self._bayesian = bool(hf.get('bayesian')[0])

                if 'batch_size' in hf.keys():
                    self._batch_size = int(hf.get('batch_size')[0])
                
                if 'hidden_sizes' in hf.keys():
                    self._hidden_sizes = tuple(hf.get('hidden_sizes')[0])

                if 'hidden_activation' in hf.keys():
                    self._hidden_activation = str(hf.get('hidden_activation')[0].decode("utf-8"))

        else:
            self._x_dim = x_dim
            self._centers = centers
            self._bayesian = bayesian
            self._batch_size = batch_size
            self._hidden_sizes = hidden_sizes
            self._hidden_activation = hidden_activation

        # create parameters dict
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

        # ask ConditionalDensityEstimator to form the MLP
        self.create_core(h5_addr = h5_addr)
        #self.core_model.model.summary()

        # create models for inference: 
        # self._prob_pred_model, self._sample_model, self._params_model, self._training_model
        self.create_models()

    def save(self, h5_addr : str) -> None:
        self.core_model.model.save(h5_addr)
        with h5py.File(h5_addr, 'a') as hf:
            hf.create_dataset('x_dim', shape=(1,len(self.x_dim)), data=self.x_dim)
            hf.create_dataset('centers', shape=(1,), data=int(self.centers))
            hf.create_dataset('bayesian', shape=(1,), data=int(self.bayesian))

            if self.batch_size is not None:
                hf.create_dataset('batch_size', shape=(1,), data=int(self.batch_size))

            if self.hidden_sizes is not None:
                hf.create_dataset('hidden_sizes', shape=(1,len(self.hidden_sizes)), data=list(self.hidden_sizes))

            if self.hidden_activation is not None:
                hf.create_dataset('hidden_activation', shape=(1,), data=str(self.hidden_activation))

    def create_models(self):

        # define X input
        self.x_input = list(self.core_model.input_slices.values())

        # put mixture components together
        self.weights = self.core_model.output_slices['mixture_weights']
        self.locs = self.core_model.output_slices['mixture_locations']
        self.scales = self.core_model.output_slices['mixture_scales']

        # create params model
        self._params_model = keras.Model(
            #inputs=self.x_input,
            inputs = {**self.core_model.input_slices},
            outputs=[
                self.weights,
                self.locs,
                self.scales,
            ],
            name="params_model",
        )

        # create prob model
        cat = tfd.Categorical(probs=self.weights,dtype=self.dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # define Y input
        self.y_input = keras.Input(
                name = "y_input",
                shape=(1),
                batch_size = self.batch_size,
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
                self.x_input,
                self.y_input,
            ],
            outputs=[
                self.pdf,
                self.log_pdf,
                self.ecdf
            ],
            name="prob_pred_model",
        )

        # pipeline training model
        self._pl_training_model = keras.Model(
            inputs={**self.core_model.input_slices, 'y_input':self.y_input},
            outputs=[
                self.log_pdf, # in shape: (batch_size,1)
            ]
        )

        # normal training model
        self._training_model = keras.Model(
            inputs=[
                self.x_input,
                self.y_input,    
            ],
            outputs=[
                self.log_pdf,
            ]
        )
        # defne the loss function
        # y_pred will be self.log_pdf which is (batch_size,1)
        self._loss = lambda y_true, y_pred: -tf.reduce_sum(y_pred)

    @property
    def centers(self):
        return self._centers

    def mean(self,
        x : npt.NDArray[np.float64],
    ):

        prediction_res = self._params_model.predict(
            x,
        )
        result_dict = {}
        for idx,param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        weights_t = tf.convert_to_tensor(result_dict['mixture_weights'], dtype=self.dtype)
        locs_t = tf.convert_to_tensor(result_dict['mixture_locations'], dtype=self.dtype)
        scales_t = tf.convert_to_tensor(result_dict['mixture_scales'], dtype=self.dtype)

        # create mixture model
        cat = tfd.Categorical(probs=weights_t,dtype=self.dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(locs_t, axis=1), tf.unstack(scales_t, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        result = mixture.mean()
        return result.numpy()


    def quantile(self,
        x, # dict as below
        samples, # numbers between 0.0 and 1.0
    ):
        """
            vectorized numerical quantile finder using Scipy.optimize.newton
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
        """
        # x = { 'queue_length1': np.zeros(1000), 'queue_length2': np.zeros(1000), 'queue_length3' : np.zeros(1000) }
        x_list = np.array([np.array([*items]) for items in zip(*x.values())])

        def model_cdf_fn(x ,a, b):
            pdf, logpdf, cdf = self.prob_batch(x=a,y=x)
            return cdf - b

        def model_pdf_fn(x, a, b=None):
            pdf, logpdf, cdf = self.prob_batch(x=a,y=x)
            return pdf

        result = optimize.newton(
            func = model_cdf_fn, 
            x0 = self.mean(x=x), # we feed the mean of the mixture as the initial guess
            args=(x_list,samples),
            fprime = model_pdf_fn,
            disp = True,
        )

        return result

    def sample_n(self, 
        x,
        rng : np.random.Generator = np.random.default_rng(seed=0),
    ):
        """
        https://github.com/tensorflow/probability/issues/659
        there is no closed form for the quantile of the Gaussian mixture
        In the last answer of these question, someone has mentioned that 
        Scipy can handle vectorized root finding:
        https://stackoverflow.com/questions/13088115/finding-the-roots-of-a-large-number-of-functions-with-one-variable
        This is what we use for quantile function.
        """
        # x = { 'queue_length1': np.zeros(1000), 'queue_length2': np.zeros(1000), 'queue_length3' : np.zeros(1000) }
        samples = rng.uniform(0.0,1.0,size = len(list(x.values())[0]))
        return self.quantile(x=x,samples=samples)
