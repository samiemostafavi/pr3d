import keras
from keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions

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