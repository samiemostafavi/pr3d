import tensorflow as tf
import keras
from keras import layers
import tensorflow_probability as tfp
import tensorflow_addons as tfa
tfd = tfp.distributions



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