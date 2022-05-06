import tensorflow as tf
import keras
from keras import layers
import numpy as np
import tensorflow_probability as tfp
import tensorflow_addons as tfa
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

        # Using functional API of keras instead of sequential
        self._input_layer = keras.Input(
                name = "x_input",
                shape=input_shape,
                batch_size=10000,
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

        # connect output layer
        # self._output_layer = layers.Concatenate()(slices)
        # print(self._output_layer)

        self._model = None

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
        
        self.cat = tfd.Categorical(logits=self.logits,dtype=tf.float64)
        self.components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(self.locs, axis=1), tf.unstack(self.scales, axis=1))]

        self.mixture = tfd.Mixture(cat=self.cat, components=self.components)

        # define pdf, logpdf, and loglikelihood
        self.pdf_ = self.mixture.prob(self.y_input)
        self.log_pdf_ = self.mixture.log_prob(self.y_input)
        self.log_likelihood_ = tf.reduce_sum(self.log_pdf_ )

        self._model = keras.Model(inputs=[self.x_input,self.y_input],outputs=[self.pdf_],name="test")
        self._model.summary()

        # define optimizer and train_step
        self.optimizer = tfa.optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=self.weight_decay, epsilon=self.epsilon)
        #self.train_step = optimizer.minimize(
        #    loss=-self.log_likelihood_,
        #    var_list=self.mlp.model.trainable_variables,
        #)

        #train = tf.keras.optimizers.Adam().minimize(
        #    loss=-self.log_likelihood_,
        #    var_list=self.mlp.model.trainable_variables,
        #    #tape=tf.GradientTape(),
        #)
     
    def fit(self, X, Y):
        
        #call the model
        mymodel = self._model()
        outputs = mymodel(X,Y)

        print(outputs)


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

    gmm_model.fit(X,Y)
    
    
    