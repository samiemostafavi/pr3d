import tensorflow as tf
import keras
import numpy as np
import numpy.typing as npt
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import h5py
from typing import Tuple
import numpy.typing as npt
import numpy as np

tfd = tfp.distributions

from pr3d.common.evm import *
from pr3d.common.core import ConditionalDensityEstimator

# in order to use tfd.Gamma.quantile
#tf.compat.v1.disable_eager_execution()

class ConditionalGammaEVM(ConditionalDensityEstimator):
    def __init__(
        self,
        x_dim : list = None,
        h5_addr : str = None,
        bayesian : bool = False,
        batch_size : int = None,
        dtype : str = 'float64',
        hidden_sizes = (16,16), 
        hidden_activation = 'tanh',
    ):

        super(ConditionalGammaEVM,self).__init__(
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
                self._bayesian = bool(hf.get('bayesian')[0])

                if 'batch_size' in hf.keys():
                    self._batch_size = int(hf.get('batch_size')[0])
                
                if 'hidden_sizes' in hf.keys():
                    self._hidden_sizes = tuple(hf.get('hidden_sizes')[0])

                if 'hidden_activation' in hf.keys():
                    self._hidden_activation = str(hf.get('hidden_activation')[0].decode("utf-8"))

        else:
            self._x_dim = x_dim
            self._bayesian = bayesian
            self._batch_size = batch_size
            self._hidden_sizes = hidden_sizes
            self._hidden_activation = hidden_activation


        # create parameters dict
        self._params_config = {
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

        # put mixture components together (from X)
        self.gamma_shape = self.core_model.output_slices['gamma_shape']
        self.gamma_rate = self.core_model.output_slices['gamma_rate']
        self.tail_param = self.core_model.output_slices['tail_parameter']
        self.tail_threshold = self.core_model.output_slices['tail_threshold']
        self.tail_scale = self.core_model.output_slices['tail_scale']

        # these models are used for printing paramters
        self._params_model = keras.Model(
            #inputs=list(self.x_input.values()),
            #inputs=self.x_input,
            inputs = {**self.core_model.input_slices},
            outputs=[
                self.gamma_shape,
                self.gamma_rate,
                self.tail_param,
                self.tail_threshold,
                self.tail_scale,
            ],
            name="params_model",
        )

        # build gamma density
        gamma = tfd.Gamma(concentration=tf.squeeze(self.gamma_shape), rate=tf.squeeze(self.gamma_rate))

        # find the normalization factor (from X)
        # squeezing the tail_threshold was important
        self.norm_factor = tf.constant(1.00,dtype=self.dtype)-gamma.cdf(tf.squeeze(self.tail_threshold))

        # define Y input
        self.y_input = keras.Input(
            name = "y_input",
            shape=(1),
            batch_size = self.batch_size,
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
        self.expanded_log_pdf = tf.expand_dims(self.log_pdf,axis=1)

        self.ecdf = tf.constant(1.00,dtype=self.dtype) - mixture_tail_prob(
            bool_split_tensor = bool_split_tensor,
            gpd_tail_prob_t = gpd_tail_prob_t,
            bulk_tail_prob_t = bulk_tail_prob_t,
            dtype = self.dtype,
        )

        # these models are used for probability predictions
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

        self.norm_factor_model = keras.Model(
            inputs=self.x_input,
            outputs=[
                tf.expand_dims(self.norm_factor, axis=0), # very important "expand_dims"
            ],
            name="norm_factor_model",
        )

        # pipeline training model
        self._pl_training_model = keras.Model(
            inputs={**self.core_model.input_slices, 'y_input':self.y_input},
            outputs=[
                self.expanded_log_pdf, # in shape: (batch_size,1)
            ]
        )

        # normal training model
        self._training_model = keras.Model(
            inputs=[
                self.x_input,
                self.y_input,
            ],
            outputs=[
                self.expanded_log_pdf, # in shape: (batch_size,1)
            ]
        )

        # defne the loss function
        # y_pred will be self.log_pdf which is (batch_size,1)
        self._loss = lambda y_true, y_pred: -tf.reduce_sum(y_pred)

    def sample_n(self, 
        x,
        rng : tf.random.Generator = tf.random.Generator.from_seed(0),
    ):

        prediction_res = self._params_model.predict(
            x,
        )
        result_dict = {}
        for idx,param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        threshold_act_t = tf.convert_to_tensor(result_dict['tail_threshold'], dtype=self.dtype)
        gamma_concentration_t = tf.convert_to_tensor(result_dict['gamma_shape'], dtype=self.dtype)
        gamma_rate_t = tf.convert_to_tensor(result_dict['gamma_rate'], dtype=self.dtype)
        gpd_scale_t = tf.convert_to_tensor(result_dict['tail_scale'],dtype=self.dtype)
        gpd_concentration_t = tf.convert_to_tensor(result_dict['tail_parameter'],dtype=self.dtype)

        samples_t = rng.uniform(
            minval = 0.00,
            maxval = 1.00,
            shape = tf.shape(gamma_concentration_t),#[n],
            dtype = self.dtype,
        )

        gamma = tfp.distributions.Gamma(
            concentration=gamma_concentration_t,#gamma_concentration.astype(dtype),
            rate=gamma_rate_t,#gamma_rate.astype(dtype),
        )

        threshold_qnt_t = gamma.cdf(threshold_act_t)

        # split the samples into bulk and tail according to the norm_factor (from X and Y)
        # gives a tensor, indicating which random_input are greater than norm_factor
        # greater than threshold is true, else false
        bool_split_t = tf.greater(samples_t, threshold_qnt_t) # this is in Boolean

        # gamma samples tensor
        gamma_samples_t = gamma.quantile(samples_t)
        
        # gpd samples tensor
        #gpd_presamples_t = tf.divide(
        #    samples_t-threshold_qnt_t,
        #    tf.constant(1.00,dtype=self.dtype)-threshold_qnt_t,
        #)

        #gpd = tfp.distributions.GeneralizedPareto(
        #    loc = 0.00,
        #    scale = gpd_scale_t,#gpd_scale.astype(dtype),
        #    concentration = gpd_concentration_t,#gpd_concentration.astype(dtype),
        #)

        #gpd_samples_t = gpd.quantile(gpd_presamples_t)/(tf.constant(1.00,dtype=self.dtype)-threshold_qnt_t)+threshold_act_t

        gpd_samples_t = gpd_quantile(
            threshold_act_t,
            gpd_concentration_t,
            gpd_scale_t,
            tf.constant(1.00,dtype=self.dtype)-threshold_qnt_t,
            samples_t,
            dtype = self.dtype,
        )

        # pass them through the mixture filter
        result = mixture_sample(
            cdf_bool_split_t = bool_split_t,
            gpd_sample_t = gpd_samples_t,
            bulk_sample_t = gamma_samples_t,
            dtype=self.dtype,
        )

        return result.numpy()