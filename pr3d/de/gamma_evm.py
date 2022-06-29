import tensorflow as tf
import keras
import tensorflow_probability as tfp
tfd = tfp.distributions
import h5py

from pr3d.common.evm import *
from pr3d.common.core import NonConditionalDensityEstimator

# in order to use tfd.Gamma.quantile
#tf.compat.v1.disable_eager_execution()

class GammaEVM(NonConditionalDensityEstimator):
    def __init__(
        self,
        h5_addr : str = None,
        bayesian : bool = False,
        batch_size : int = None,
        dtype : str = 'float64',
    ):
        super(GammaEVM,self).__init__(
            h5_addr = h5_addr,
            bayesian = bayesian,
            batch_size = batch_size,
            dtype = dtype,
        )
        
        # figure out parameters
        if h5_addr is not None:
            # read side parameters
            with h5py.File(h5_addr, 'r') as hf:
                self._bayesian = bool(hf.get('bayesian')[0])

                if 'batch_size' in hf.keys():
                    self._batch_size = int(hf.get('batch_size')[0])
        else:
            self._bayesian = bayesian
            self._batch_size = batch_size
        

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
            }
        }

        # ask NonConditionalDensityEstimator to form the SLP
        self.create_core(h5_addr = h5_addr)
        #self._core_model.model.summary()

        # create models for inference: 
        # self._prob_pred_model, self._sample_model, self._params_model, self._training_model
        self.create_models()

    def save(self, h5_addr : str) -> None:
        self.core_model.model.save(h5_addr)
        with h5py.File(h5_addr, 'a') as hf:
            hf.create_dataset('bayesian', shape=(1,), data=int(self.bayesian))
            if self.batch_size is not None:
                hf.create_dataset('batch_size', shape=(1,), data=int(self.batch_size))

    def create_models(self):

        # now lets define the models to get probabilities

        # define dummy input
        self.dummy_input = self.core_model.input_layer

        # put mixture components together (from X)
        self.gamma_shape = self.core_model.output_slices['gamma_shape']
        self.gamma_rate = self.core_model.output_slices['gamma_rate']
        self.tail_param = self.core_model.output_slices['tail_parameter']
        self.tail_threshold = self.core_model.output_slices['tail_threshold']
        self.tail_scale = self.core_model.output_slices['tail_scale']

        # these models are used for printing paramters
        self._params_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                self.gamma_shape,
                self.gamma_rate,
                self.tail_param,
                self.tail_threshold,
                self.tail_scale,
            ],
            name="params_model",
        )

        # build gamma density (from X)
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

        self.norm_factor_model = keras.Model(
            inputs=self.dummy_input,
            outputs=[
                tf.expand_dims(self.norm_factor, axis=0), # very important "expand_dims"
            ],
            name="norm_factor_model",
        )

        # pipeline training model
        self._pl_training_model = keras.Model(
            inputs={'dummy_input':self.dummy_input, 'y_input':self.y_input},
            outputs=[
                self.expanded_log_pdf, # in shape: (batch_size,1)
            ]
        )

        # normal training model
        self._training_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.y_input,    
            ],
            outputs=[
                self.expanded_log_pdf, # in shape: (batch_size,1)
            ]
        )

        # defne the loss function
        # y_pred will be self.log_pdf which is (batch_size,1)
        self._loss = lambda y_true, y_pred: -tf.reduce_sum(y_pred)

        # create the sampling model
        # sample_input: random uniform numbers in [0,1]
        # feed them to the inverse cdf of the distribution

        # we use the threshold in CDF domain which is norm_factor and create cdf_bool_split_t
        # split sample_input into the ones greater or smaller than norm_factor
        # feed smallers to the icdf of Gamma, feed larger values to the icdf of GPD


        '''
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
        # ONLY WORKS WITH tf.compat.v1.disable_eager_execution()
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

        self._sample_model = keras.Model(
            inputs=[
                self.dummy_input,
                self.sample_input,
            ],
            outputs=[
                self.sample,
            ],
            name="sample_model",
        )
        '''
