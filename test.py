import tensorflow as tf
import keras
from keras import layers
import numpy as np
import numpy.typing as npt
import tensorflow_probability as tfp
import tensorflow_addons as tfa
from typing import Tuple
tfd = tfp.distributions

print('TensorFlow version: {}'.format(tf.__version__))


class MLP():
    def __init__(self,
        name : str,
        input_shape : int,
        output_layer_config : dict,
        dtype,
        hidden_sizes, 
        hidden_activation,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        dropout_ph=None
    ):
        """
        :param dropout_ph: None if no dropout should be used. Else a scalar placeholder that determines the prob of dropping a node.
        Remember to set placeholder to Zero during test / eval
        """
        self.output_layer_config = output_layer_config

        # Using functional API of keras instead of sequential
        self._input_layer = keras.Input(
                name = "x_input",
                shape=input_shape,
                #batch_size=10000,
                dtype=dtype,
        )

        for idx, hidden_size in enumerate(hidden_sizes):
            if idx == 0:
                prev_layer = self._input_layer

            # create the new hidden layer
            hidden_layer = layers.Dense(
                name="hidden_%d" % idx,
                units=hidden_size,
                activation=hidden_activation,
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                dtype=dtype,
            )

            # connect the new layer
            prev_layer = hidden_layer(prev_layer)

            # create and connect the dropout layer
            if dropout_ph is not None:
                dropout_layer = layers.Dropout(dropout_ph)
                prev_layer = dropout_layer(prev_layer)

        # create output layer by concatenating slices
        self._output_slices = {}
        slices = []
        for slice_name in output_layer_config:
            slice_dense = layers.Dense(
                name="dense_"+slice_name,
                units=output_layer_config[slice_name]['slice_size'],
                activation=output_layer_config[slice_name]['slice_activation'],
                kernel_initializer=kernel_initializer,
                bias_initializer=bias_initializer,
                dtype=dtype,
            )
            self._output_slices[slice_name] = slice_dense(prev_layer)
            slices.append(self._output_slices[slice_name])

        # connect output layer
        self._output_layer = layers.Concatenate()(slices)
        # print(self._output_layer)

        # create model
        self._model = keras.Model(inputs=self._input_layer,outputs=self._output_layer,name="mlp")
        self._model.summary()

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def output_slices(self):
        return self._output_slices

    @property
    def model(self):
        return self._model



class ConditionalGMM():
    def __init__(
        self,
        centers : int = 8,
        x_dim : int = 3,
        hidden_sizes : tuple = (16,16),
        learning_rate : float = 5e-3,
        weight_decay : float = 0.0,
        epsilon : float = 1e-8,
    ):
        self.centers = centers
        self.x_dim = x_dim
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate 
        self.weight_decay = weight_decay
        self.epsilon = epsilon
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

    def negative_log_likelihood_loss(self, y_true, y_pred):

        # y_pred is the concatenated mixture_weights, mixture_locations, and mixture_scales
        logits = y_pred[:,0:self.centers-1]
        locs = y_pred[:,self.centers:2*self.centers-1]
        scales = y_pred[:,2*self.centers:3*self.centers-1]
        
        cat = tfd.Categorical(logits=logits,dtype=tf.float64)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(locs, axis=1), tf.unstack(scales, axis=1))]

        mixture = tfd.Mixture(cat=cat, components=components)

        # define logpdf and loglikelihood
        log_pdf_ = mixture.log_prob(y_true)
        log_likelihood_ = tf.reduce_sum(log_pdf_ )
        return -log_likelihood_
        

    def create_model(self):
        self.mlp = MLP(
            name = 'gmm_keras_model',
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
            },
            hidden_sizes=self.hidden_sizes,
            hidden_activation='tanh',
            dtype=tf.dtypes.float64,
        )
        #self.mlp.model.summary()    

        # define optimizer and train_step
        self.optimizer = tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay, epsilon=self.epsilon)

        # this keras model is the one that we use for training
        self.mlp.model.compile(optimizer=self.optimizer, loss=self.negative_log_likelihood_loss)

        # now lets define the models to get probabilities

        # define Y input
        self.y_input = keras.Input(
                name = "y_input",
                shape=(1),
                dtype=tf.float64,
        )

        # define X input
        self.x_input = self.mlp.input_layer

        # put mixture components together
        self.logits = self.mlp.output_slices['mixture_weights']
        self.locs = self.mlp.output_slices['mixture_locations']
        self.scales = self.mlp.output_slices['mixture_scales']
        
        cat = tfd.Categorical(logits=self.logits,dtype=tf.float64)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]
        mixture = tfd.Mixture(cat=cat, components=components)

        # define pdf, logpdf and loglikelihood
        self.pdf = mixture.prob(self.y_input)
        self.log_pdf = mixture.log_prob(self.y_input)
        self.ecdf = mixture.cdf(self.y_input)

        # these models are used for probability predictions
        self.theta_pred_model = keras.Model(inputs=self.x_input,outputs=[self.logits,self.locs,self.scales],name="theta_pred_model")
        self.prob_pred_model = keras.Model(inputs=[self.x_input,self.y_input],outputs=[self.pdf,self.log_pdf,self.ecdf],name="prob_pred_model")
     
    def fit(self, X, Y):
        
        history = self.mlp.model.fit(
            X,
            Y,
            batch_size=1000,
            epochs=100,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            #validation_data=(x_val, y_val),
        )


def create_dataset(n_samples = 300, x_dim=3, x_max = 10, x_level=2):

    # generate random sample, two components
    np.random.seed(0)

    X = np.array(np.random.randint(x_max, size=(n_samples, x_dim))*x_level)

    Y = np.array([ 
            np.random.normal(loc=x_sample[0]+x_sample[1]+x_sample[2],scale=(x_sample[0]+x_sample[1]+x_sample[2])/5) 
                for x_sample in X 
        ]
    )

    return X,Y

if __name__ == "__main__":

    """
    x_dim = 3

    emm_centers = 2
    emm_model = MLP(
        name = 'test',
        input_shape=(x_dim),
        output_layer_config={
            'mixture_locations': { 
                'slice_size' : emm_centers,
                'slice_activation' : None,
            },
            'mixture_scales': { 
                'slice_size' : emm_centers,
                'slice_activation' : 'softplus',
            },
            'mixture_weights': { 
                'slice_size' : emm_centers,
                'slice_activation' : 'softmax',
            },
            'tail_param' : { 
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
            },
        },
        hidden_sizes=(16,16),
        hidden_activation='tanh',
        dtype=tf.dtypes.float64,
    )
    emm_model.model.summary()
    """

    gmm_model = ConditionalGMM()

    X,Y = create_dataset(n_samples = 10000, x_dim = 3)
    print("X shape: {0}".format(X.shape))
    print("Y shape: {0}".format(Y.shape))

    # train the model
    gmm_model.fit(X,Y)

    print("Single test x: {0}, and y: {1}".format(X[10,:],Y[10]))
    pdf,log_pdf,ecdf = gmm_model.prob_single(X[10,:],Y[10])
    print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))

    print("Batch test x: {0}, and y: {1}".format(X[10:15,:],Y[10:15]))
    pdf,log_pdf,ecdf = gmm_model.prob_batch(X[10:15,:],Y[10:15])
    print("Result pdf: {0}, log_pdf: {1}, cdf: {2}".format(pdf,log_pdf,ecdf))
    
    
    