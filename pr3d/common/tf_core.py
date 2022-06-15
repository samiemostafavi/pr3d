import tensorflow as tf
import keras
from keras import layers
import numpy as np
import numpy.typing as npt
from typing import Tuple
import tensorflow_addons as tfa

class MLP():

    def __init__(
        self,
        name : str = 'mlp',
        input_shape = (3),
        output_layer_config : dict = None,
        dtype = tf.dtypes.float64,
        hidden_sizes = (16,16), 
        hidden_activation = 'tanh',
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        dropout_ph=None,
        loaded_mlp_model : keras.Model = None,
        batch_size = None,
    ):
        """
        :param dropout_ph: None if no dropout should be used. Else a scalar placeholder that determines the prob of dropping a node.
        Remember to set placeholder to Zero during test / eval
        """

        if loaded_mlp_model is not None:
            # set the model
            self._model = loaded_mlp_model

            # find the input and output layers
            self._input_layer = self._model.input
            self._output_layer = self._model.get_layer('output')

            # find slice layer names
            slice_names = []
            int_node = self._output_layer._inbound_nodes[0]
            for idx, layer in enumerate(int_node.inbound_layers):
                slice_names.append(layer.name)

            # create output layer by concatenating slices
            self._output_slices = {}
            for slice_name in slice_names:
                self._output_slices[slice_name] = self._model.get_layer(slice_name).output

        else:
            # Using functional API of keras instead of sequential
            self._input_layer = keras.Input(
                    name = "input",
                    shape=input_shape,
                    batch_size=batch_size,
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
                    name=slice_name,
                    units=output_layer_config[slice_name]['slice_size'],
                    activation=output_layer_config[slice_name]['slice_activation'],
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    dtype=dtype,
                )
                self._output_slices[slice_name] = slice_dense(prev_layer)
                slices.append(self._output_slices[slice_name])

            # connect output layer
            self._output_layer = layers.Concatenate(name='output')(slices)
            # print(self._output_layer)

            # create model
            self._model = keras.Model(inputs=self._input_layer,outputs=self._output_layer,name=name)
            #self._model.summary()

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def output_slices(self):
        return self._output_slices

    @property
    def model(self):
        return self._model


# Single Layer Perceptron
class SLP():

    def __init__(
        self,
        name : str = 'slp',
        layer_config : dict = None,
        dtype = tf.dtypes.float64,
        kernel_initializer = 'glorot_uniform',
        bias_initializer = 'zeros',
        loaded_slp_model : keras.Model = None,
        batch_size = None,
    ):
        """
        Remember to set placeholder to Zero during test / eval
        """

        if loaded_slp_model is not None:
            # set the model
            self._model = loaded_slp_model

            # find the input and output layers
            self._input_layer = self._model.input
            self._output_layer = self._model.get_layer('output')

            # find slice layer names
            slice_names = []
            int_node = self._output_layer._inbound_nodes[0]
            for idx, layer in enumerate(int_node.inbound_layers):
                slice_names.append(layer.name)

            # create output layer by concatenating slices
            self._output_slices = {}
            for slice_name in slice_names:
                self._output_slices[slice_name] = self._model.get_layer(slice_name).output

        else:

            # Using functional API of keras

            # This is just a dummy input
            self._input_layer = keras.Input(
                    name = "input",
                    shape=(1),
                    batch_size=batch_size,
                    dtype=dtype,
            )

            # create the single layer by concatenating slices
            self._output_slices = {}
            slices = []
            for slice_name in layer_config:
                slice_dense = layers.Dense(
                    name=slice_name,
                    units=layer_config[slice_name]['slice_size'],
                    activation=layer_config[slice_name]['slice_activation'],
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    dtype=dtype,
                )
                self._output_slices[slice_name] = slice_dense(self._input_layer)
                slices.append(self._output_slices[slice_name])

            # connect output layer
            self._output_layer = layers.Concatenate(name='output')(slices)
            # print(self._output_layer)

            # create model
            self._model = keras.Model(inputs=self._input_layer,outputs=self._output_layer,name=name)
            #self._model.summary()


    @property
    def input_layer(self):
        return self._input_layer

    @property
    def output_slices(self):
        return self._output_slices

    @property
    def model(self):
        return self._model



class NonConditionalDensityEstimator(): 
    def __init__(
        self,
        h5_addr : str = None,
        dtype : str = 'float64',
    ):
        pass

    def save(self, h5_addr : str):
        pass

    def prob_single(self, y : np.float64) -> Tuple[np.float64, np.float64, np.float64]:

        # for single value x (not batch)
        # y : np.float64 number
        x = 0
        [pdf,log_pdf,ecdf] = self.prob_pred_model.predict(
            [np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)],
        )
        return np.squeeze(pdf),np.squeeze(log_pdf),np.squeeze(ecdf)

    def prob_batch(self, 
        y : npt.NDArray[np.float64],
        batch_size=None,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,    
    ):
        
        # for large batches of input y
        # y : np.array of np.float64 with the shape (batch_size,1) e.g. np.array([5,6,7,8,9,10])
        x = np.zeros(len(y))
        [pdf,log_pdf,ecdf] = self.prob_pred_model.predict(
            [x,y],
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return np.squeeze(pdf),np.squeeze(log_pdf),np.squeeze(ecdf)

    def sample_n(self, 
        n : int, 
        random_generator: np.random.Generator = np.random.default_rng(),
        batch_size=None,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ) -> npt.NDArray[np.float64]:

        # generate n random numbers uniformly distributed on [0,1]
        x = np.zeros(n)
        y = random_generator.uniform(0,1,n)

        samples = self.sample_model.predict(
            [x,y],
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return np.squeeze(samples)

    def get_parameters(self) -> dict:

        # for single value x (not batch)
        # y : np.float64 number
        x = 0
        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        prediction_res = self.params_model.predict(np.expand_dims(x, axis=0))

        result_dict = {}
        for idx,param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        return result_dict

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
        self.slp_model.model.compile(optimizer=optimizer, loss=self.loss)

        X = np.zeros(len(Y))
        history = self.slp_model.model.fit(
            X,
            Y,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            #validation_data=(x_val, y_val),
        )

    @property
    def prob_pred_model(self) -> keras.Model:
        return self._prob_pred_model

    @property
    def sample_model(self) -> keras.Model:
        return self._sample_model

    @property
    def params_model(self) -> keras.Model:
        return self._params_model

    @property
    def params_config(self) -> dict:
        return self._params_config

    @property
    def slp_model(self) -> SLP:
        return self._slp_model

    @property
    def loss(self):
        return self._loss