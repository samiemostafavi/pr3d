import keras
import numpy as np
import tensorflow as tf
from keras import layers

from pr3d.common.bayesian import SavableDenseFlipout


def create_model_inputs(feature_names, dtype=tf.float32):
    inputs = {}
    for feature_name in feature_names:
        inputs[feature_name] = layers.Input(
            name=feature_name,
            shape=(1,),
            dtype=dtype,
        )
    return inputs

def create_recurrent_model_inputs(feature_names, recurrent_taps, batch_size, dtype=tf.float32):
    inputs = {}
    for feature_name in feature_names:
        inputs[feature_name] = layers.Input(
            name=feature_name,
            shape=(recurrent_taps,1),
            dtype=dtype,
            batch_size=batch_size,
        )
    return inputs

def squeeze_generic(a, axes_to_keep):
    out_s = [s for i, s in enumerate(a.shape) if i in axes_to_keep or s != 1]
    return a.reshape(out_s)


class RnnMLP:
    def __init__(
        self,
        batch_size,
        feature_names: list = None,  # list(str)
        name: str = None,
        recurrent_taps: int = None,
        hidden_layers_config: dict = None,
        output_layer_config: dict = None,
        dtype=None,
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",  # 'zeros'
        dropout_ph=None,
        loaded_mlp_model: keras.Model = None,
    ):
        """
        :param dropout_ph: None if no dropout should be used. Else a scalar placeholder that determines the prob of dropping a node.
        Remember to set placeholder to Zero during test / eval
        """

        if loaded_mlp_model is not None:
            # set the model
            self._model = loaded_mlp_model

            # find the input layer and slices
            self._input_layer = self._model.get_layer("input")
            # find slice layer names
            slice_names = []
            int_node = self._input_layer._inbound_nodes[0]

            # figure out inbound_layers size
            tmp = np.array([int_node.inbound_layers])
            tmp = np.transpose(tmp)
            tmp = squeeze_generic(tmp, [0])
            if len(tmp) > 1:
                # more than one input layer
                for idx, layer in enumerate(int_node.inbound_layers):
                    slice_names.append(layer.name)
            else:
                # only one input layer
                slice_names.append(int_node.inbound_layers.name)

            # create input slices by concatenating slices
            self._input_slices = {}
            for slice_name in slice_names:
                self._input_slices[slice_name] = self._model.get_layer(slice_name).input

            # find the output layer and slices
            self._output_layer = self._model.get_layer("output")
            # find slice layer names
            slice_names = []
            int_node = self._output_layer._inbound_nodes[0]
            for idx, layer in enumerate(int_node.inbound_layers):
                slice_names.append(layer.name)
            # create output layer by concatenating slices
            self._output_slices = {}
            for slice_name in slice_names:
                self._output_slices[slice_name] = self._model.get_layer(
                    slice_name
                ).output

        else:
            
            if len(feature_names) < 2:
                raise Exception("recurrent MLP must have minimum 2 features: one for the target sequence, one for an actual feature")

            self._input_slices = create_recurrent_model_inputs(feature_names,recurrent_taps,batch_size,dtype)
            self._input_layer = keras.layers.concatenate(
                list(self._input_slices.values()), 
                name="input",
                axis=2
            )

            # features = layers.BatchNormalization()(features)

            for idx, hidden_layer_name in enumerate(hidden_layers_config):
                if idx == 0:
                    prev_layer = self._input_layer

                # create the new hidden layer
                if hidden_layers_config[hidden_layer_name]["type"] == "lstm":
                    hidden_layer = layers.LSTM(
                        name=hidden_layer_name,
                        units=hidden_layers_config[hidden_layer_name]["size"],
                        return_sequences=hidden_layers_config[hidden_layer_name]["return_sequences"],
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        dtype=dtype,
                    )
                elif hidden_layers_config[hidden_layer_name]["type"] == "dense":
                    hidden_layer = layers.Dense(
                        name=hidden_layer_name,
                        units=hidden_layers_config[hidden_layer_name]["size"],
                        activation=hidden_layers_config[hidden_layer_name]["activation"],
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        dtype=dtype,
                    )
                else:
                    raise Exception("wrong hidden layer type")

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
                    units=output_layer_config[slice_name]["slice_size"],
                    activation=output_layer_config[slice_name]["slice_activation"],
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    dtype=dtype,
                )
                self._output_slices[slice_name] = slice_dense(prev_layer)
                slices.append(self._output_slices[slice_name])

            # connect output layer
            self._output_layer = layers.Concatenate(name="output")(slices)
            # print(self._output_layer)

            # create model
            self._model = keras.Model(
                inputs=self._input_slices, outputs=self._output_layer, name=name
            )
            self._model.summary()

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def input_slices(self):
        return self._input_slices

    @property
    def output_slices(self):
        return self._output_slices

    @property
    def model(self):
        return self._model



class MLP:
    def __init__(
        self,
        bayesian,
        batch_size,
        feature_names: list,  # list(str)
        name: str = "mlp",
        output_layer_config: dict = None,
        dtype=tf.dtypes.float64,
        hidden_sizes=(16, 16),
        hidden_activation="tanh",
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",  # 'zeros'
        dropout_ph=None,
        loaded_mlp_model: keras.Model = None,
    ):
        """
        :param dropout_ph: None if no dropout should be used. Else a scalar placeholder that determines the prob of dropping a node.
        Remember to set placeholder to Zero during test / eval
        """

        if loaded_mlp_model is not None:
            # set the model
            self._model = loaded_mlp_model

            # find the input layer and slices
            self._input_layer = self._model.get_layer("input")
            # find slice layer names
            slice_names = []
            int_node = self._input_layer._inbound_nodes[0]

            # figure out inbound_layers size
            tmp = np.array([int_node.inbound_layers])
            tmp = np.transpose(tmp)
            tmp = squeeze_generic(tmp, [0])
            if len(tmp) > 1:
                # more than one input layer
                for idx, layer in enumerate(int_node.inbound_layers):
                    slice_names.append(layer.name)
            else:
                # only one input layer
                slice_names.append(int_node.inbound_layers.name)

            # create input slices by concatenating slices
            self._input_slices = {}
            for slice_name in slice_names:
                self._input_slices[slice_name] = self._model.get_layer(slice_name).input

            # find the output layer and slices
            self._output_layer = self._model.get_layer("output")
            # find slice layer names
            slice_names = []
            int_node = self._output_layer._inbound_nodes[0]
            for idx, layer in enumerate(int_node.inbound_layers):
                slice_names.append(layer.name)
            # create output layer by concatenating slices
            self._output_slices = {}
            for slice_name in slice_names:
                self._output_slices[slice_name] = self._model.get_layer(
                    slice_name
                ).output

        else:

            # Using functional API of keras instead of sequential
            self._input_slices = create_model_inputs(feature_names)
            if len(feature_names) == 1:
                self._input_layer = layers.Dense(
                    1, activation=None, use_bias=False, name="input"
                )(list(self._input_slices.values())[0])
            else:
                self._input_layer = keras.layers.concatenate(
                    list(self._input_slices.values()), name="input"
                )

            # features = layers.BatchNormalization()(features)

            for idx, hidden_size in enumerate(hidden_sizes):
                if idx == 0:
                    prev_layer = self._input_layer

                # create the new hidden layer
                if bayesian:
                    hidden_layer = SavableDenseFlipout(
                        name="hidden_%d" % idx,
                        units=hidden_size,
                        activation=hidden_activation,
                        batch_size=batch_size,
                        dtype=dtype,
                    )
                else:
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
                if bayesian:
                    slice_dense = SavableDenseFlipout(
                        name=slice_name,
                        units=output_layer_config[slice_name]["slice_size"],
                        activation=output_layer_config[slice_name]["slice_activation"],
                        batch_size=batch_size,
                        dtype=dtype,
                    )
                else:
                    slice_dense = layers.Dense(
                        name=slice_name,
                        units=output_layer_config[slice_name]["slice_size"],
                        activation=output_layer_config[slice_name]["slice_activation"],
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        dtype=dtype,
                    )
                self._output_slices[slice_name] = slice_dense(prev_layer)
                slices.append(self._output_slices[slice_name])

            # connect output layer
            self._output_layer = layers.Concatenate(name="output")(slices)
            # print(self._output_layer)

            # create model
            self._model = keras.Model(
                inputs=self._input_slices, outputs=self._output_layer, name=name
            )
            # self._model.summary()

    @property
    def input_layer(self):
        return self._input_layer

    @property
    def input_slices(self):
        return self._input_slices

    @property
    def output_slices(self):
        return self._output_slices

    @property
    def model(self):
        return self._model


# Single Layer Perceptron
class SLP:
    def __init__(
        self,
        bayesian,
        batch_size,
        name: str = "slp",
        layer_config: dict = None,
        dtype=tf.dtypes.float64,
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",  # 'zeros' VERY IMPORTANT! Otherwise all components would be the same
        loaded_slp_model: keras.Model = None,
    ):
        """
        Remember to set placeholder to Zero during test / eval
        """

        if loaded_slp_model is not None:
            # set the model
            self._model = loaded_slp_model

            # find the input and output layers
            self._input_layer = self._model.input
            self._output_layer = self._model.get_layer("output")

            # find slice layer names
            slice_names = []
            int_node = self._output_layer._inbound_nodes[0]
            for idx, layer in enumerate(int_node.inbound_layers):
                slice_names.append(layer.name)

            # create output layer by concatenating slices
            self._output_slices = {}
            for slice_name in slice_names:
                self._output_slices[slice_name] = self._model.get_layer(
                    slice_name
                ).output

        else:

            # Using functional API of keras

            # This is just a dummy input
            self._input_layer = keras.Input(
                name="input",
                shape=(1),
                batch_size=batch_size,
                dtype=dtype,
            )

            # create the single layer by concatenating slices
            self._output_slices = {}
            slices = []
            for slice_name in layer_config:
                if bayesian:
                    slice_dense = SavableDenseFlipout(
                        name=slice_name,
                        units=layer_config[slice_name]["slice_size"],
                        activation=layer_config[slice_name]["slice_activation"],
                        batch_size=batch_size,
                        dtype=dtype,
                    )
                else:
                    slice_dense = layers.Dense(
                        name=slice_name,
                        units=layer_config[slice_name]["slice_size"],
                        activation=layer_config[slice_name]["slice_activation"],
                        kernel_initializer=kernel_initializer,
                        bias_initializer=bias_initializer,
                        dtype=dtype,
                    )
                self._output_slices[slice_name] = slice_dense(self._input_layer)
                slices.append(self._output_slices[slice_name])

            # connect output layer
            self._output_layer = layers.Concatenate(name="output")(slices)
            # print(self._output_layer)

            # create model
            self._model = keras.Model(
                inputs=self._input_layer, outputs=self._output_layer, name=name
            )
            # self._model.summary()

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
class RnnSLP:
    def __init__(
        self,
        batch_size,
        name: str = None,
        recurrent_taps: int = None,
        layer_config: dict = None,
        dtype = None,
        kernel_initializer="glorot_uniform",
        bias_initializer="glorot_uniform",  # 'zeros' VERY IMPORTANT! Otherwise all components would be the same
        loaded_slp_model: keras.Model = None,
    ):
        """
        Remember to set placeholder to Zero during test / eval
        """

        if loaded_slp_model is not None:
            # set the model
            self._model = loaded_slp_model

            # find the input and output layers
            self._input_layer = self._model.input
            self._output_layer = self._model.get_layer("output")

            # find slice layer names
            slice_names = []
            int_node = self._output_layer._inbound_nodes[0]
            for idx, layer in enumerate(int_node.inbound_layers):
                slice_names.append(layer.name)

            # create output layer by concatenating slices
            self._output_slices = {}
            for slice_name in slice_names:
                self._output_slices[slice_name] = self._model.get_layer(
                    slice_name
                ).output

        else:

            # Using functional API of keras
            # create a sequential model where you have multiple pairs of LSTM and Dense layers, and then concatenate all the Dense layers.

            # Define the input shape for the LSTM layers
            input_shape = (recurrent_taps, 1)

            # Create an input layer
            self._input_layer = keras.Input(
                name="input",
                shape=input_shape,
                batch_size=batch_size,
                dtype=dtype,
            )

            # create the single layer by concatenating slices
            self._output_slices = {}
            slices = []
            for slice_name in layer_config:
                # When defining your LSTM model in Keras, the batch_size parameter is often not explicitly set in the layer definition.
                lstm_slice_name = slice_name + "_lstm"
                slice_lstm = layers.LSTM(
                    name=lstm_slice_name,
                    units=layer_config[slice_name]["slice_size"],
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    dtype=dtype,
                )(self._input_layer)

                slice_dense = layers.Dense(
                    name=slice_name,
                    units=layer_config[slice_name]["slice_size"],
                    activation=layer_config[slice_name]["slice_activation"],
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer,
                    dtype=dtype,
                )(slice_lstm)                    

                self._output_slices[slice_name] = slice_dense
                slices.append(slice_dense)

            # connect output layer
            self._output_layer = layers.Concatenate(name="output")(slices)
            # print(self._output_layer)

            # create model
            self._model = keras.Model(
                inputs=self._input_layer, outputs=self._output_layer, name=name
            )
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
    
