from typing import Tuple

import keras
import numpy as np
import numpy.typing as npt
import tensorflow as tf

from pr3d.common.bayesian import SavableDenseFlipout
from pr3d.common.tf import MLP, SLP, RnnMLP, RnnSLP


class DensityEstimator:
    def __init__(
        self,
        h5_addr: str = None,
        bayesian: bool = False,
        batch_size: int = None,
        dtype: str = "float64",
    ):
        self._bayesian = bayesian
        self._batch_size = batch_size

        # configure keras to use dtype
        tf.keras.backend.set_floatx(dtype)

        # for creating the tensors
        if dtype == "float64":
            self._dtype = tf.float64
        elif dtype == "float32":
            self._dtype = tf.float32
        elif dtype == "float16":
            self._dtype = tf.float16
        else:
            raise Exception("unknown dtype format")

    def create_core(self, h5_addr: str):
        pass

    def save(self, h5_addr: str):
        pass

    def create_models(self):
        pass

    def prob_single(self):
        pass

    def prob_batch(self):
        pass

    def sample_n(self):
        pass

    def get_parameters(self):
        pass

    def fit(self):
        pass

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
    def training_model(self) -> keras.Model:
        return self._training_model

    @property
    def pl_training_model(self) -> keras.Model:
        return self._pl_training_model

    @property
    def params_config(self) -> dict:
        return self._params_config

    @property
    def core_model(self):
        return self._core_model

    @property
    def loss(self):
        return self._loss

    @property
    def bayesian(self):
        return self._bayesian

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def dtype(self):
        return self._dtype


class NonConditionalDensityEstimator(DensityEstimator):
    def create_core(self, h5_addr: str):

        # initiate the slp model
        if h5_addr is not None:
            # load the keras model and feed to SLP
            if self.bayesian:
                self._core_model = SLP(
                    loaded_slp_model=keras.models.load_model(
                        h5_addr,
                        custom_objects={
                            "SavableDenseFlipout": SavableDenseFlipout,
                        },
                    ),
                    bayesian=self.bayesian,
                    batch_size=self.batch_size,
                )
            else:
                self._core_model = SLP(
                    loaded_slp_model=keras.models.load_model(
                        h5_addr,
                    ),
                    bayesian=self.bayesian,
                    batch_size=self.batch_size,
                )

        else:
            # create SLP model
            self._core_model = SLP(
                name="slp_keras_model",
                bayesian=self.bayesian,
                batch_size=self.batch_size,
                layer_config=self.params_config,
                dtype=self.dtype,
            )

    def prob_single(self, y: np.float64) -> Tuple[np.float64, np.float64, np.float64]:

        # for single value x (not batch)
        # y : np.float64 number
        x = 0
        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)],
        )
        return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def prob_batch(
        self,
        y: npt.NDArray[np.float64],
        batch_size=32,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):

        # for large batches of input y
        # y : np.array of np.float64 with the shape (batch_size,1) e.g. np.array([5,6,7,8,9,10])
        x = np.zeros(len(y))

        # IMPORTANT: batch size by default in keras is set to 32, if data length is 32*k+1, it raises error.
        if len(y) > batch_size:
            if len(y) % batch_size == 1:
                batch_size = batch_size * 2

        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [x, y],
            batch_size=len(y),
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def sample_n(
        self,
        n: int,
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
        y = random_generator.uniform(0, 1, n)

        samples = self.sample_model.predict(
            [x, y],
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
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        return result_dict

    def fit(
        self,
        Y,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.training_model.compile(optimizer=optimizer, loss=self.loss)

        X = np.zeros(len(Y))
        # history = self.core_model.model.fit(
        self.training_model.fit(
            x=[X, Y],
            y=Y,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            # validation_data=(x_val, y_val),
        )

    def fit_pipeline(
        self,
        train_dataset,
        test_dataset,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.pl_training_model.compile(optimizer=optimizer, loss=self.loss)

        # In this train_dataset, there must be an all zero column

        # history = self.core_model.model.fit(
        self.pl_training_model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=test_dataset,
            # metrics=[keras.metrics.KLDivergence()]
        )



class NonConditionalRecurrentDensityEstimator(DensityEstimator):

    def __init__(
        self,
        recurrent_taps: int = None,
        h5_addr: str = None,
        batch_size: int = None,
        dtype: str = None,
    ):

        super(NonConditionalRecurrentDensityEstimator, self).__init__(
            h5_addr=h5_addr,
            batch_size=batch_size,
            dtype=dtype,
        )

        self._recurrent_taps = recurrent_taps

    def create_core(self, h5_addr: str):

        # initiate the slp model
        if h5_addr is not None:
            # load the keras model and feed to SLP
            self._core_model = RnnSLP(
                loaded_slp_model=keras.models.load_model(
                    h5_addr,
                ),
                batch_size=self.batch_size,
                recurrent_taps=self.recurrent_taps,
            )
        else:
            # create SLP model
            self._core_model = RnnSLP(
                name="rnnslp_keras_model",
                batch_size=self.batch_size,
                recurrent_taps=self.recurrent_taps,
                layer_config=self.params_config,
                dtype=self.dtype,
            )

    def prob_single(
        self, 
        Y: npt.NDArray[np.float64], 
        y: np.float64
    ) -> Tuple[np.float64, np.float64, np.float64]:
        
        # for single value x (not batch)
        # y : np.float64 number
        # Y : np array with length equal to self.recurrent_taps
        if len(Y) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [np.expand_dims(Y, axis=0), np.expand_dims(y, axis=0)],
        )
        return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def prob_batch(
        self,
        Y: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        batch_size=32,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        # for large batches of input y
        # y : np.array of np.float64 with the shape (batch_size,1) e.g. np.array([5,6,7,8,9,10])
        # Y : 2D np array with the shape (batch_size, self.recurrent_taps)
        x = np.zeros(len(y))

        # IMPORTANT: batch size by default in keras is set to 32, if data length is 32*k+1, it raises error.
        if len(y) > batch_size:
            if len(y) % batch_size == 1:
                batch_size = batch_size * 2

        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [Y, y],
            batch_size=len(y),
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def get_parameters(self, Y: npt.NDArray[np.float64]) -> dict:

        # for a single sequence (not batch)
        # Y : np array with length equal to self.recurrent_taps
        if len(Y) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        prediction_res = self.params_model.predict(np.expand_dims(Y, axis=0))

        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        return result_dict

    def fit(
        self,
        Y,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.training_model.compile(optimizer=optimizer, loss=self.loss)

        X = np.zeros(len(Y))
        # history = self.core_model.model.fit(
        self.training_model.fit(
            x=[X, Y],
            y=Y,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            # validation_data=(x_val, y_val),
        )

    def fit_pipeline(
        self,
        train_dataset,
        test_dataset,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.pl_training_model.compile(optimizer=optimizer, loss=self.loss)

        # In this train_dataset, there must be an all zero column

        # history = self.core_model.model.fit(
        self.pl_training_model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=test_dataset,
            # metrics=[keras.metrics.KLDivergence()]
        )

    @property
    def recurrent_taps(self):
        return self._recurrent_taps


class ConditionalDensityEstimator(DensityEstimator):
    def __init__(
        self,
        x_dim: int = None,
        h5_addr: str = None,
        bayesian: bool = False,
        batch_size: int = None,
        dtype: str = "float64",
        hidden_sizes=(16, 16),
        hidden_activation="tanh",
    ):
        self._x_dim = x_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_activation = hidden_activation
        super(ConditionalDensityEstimator, self).__init__(
            h5_addr=h5_addr,
            bayesian=bayesian,
            batch_size=batch_size,
            dtype=dtype,
        )

    def create_core(self, h5_addr: str):

        # initiate the slp model
        if h5_addr is not None:
            # load the keras model and feed to MLP
            if self.bayesian:
                self._core_model = MLP(
                    loaded_mlp_model=keras.models.load_model(
                        h5_addr,
                        custom_objects={
                            "SavableDenseFlipout": SavableDenseFlipout,
                        },
                    ),
                    feature_names=self.x_dim,
                    hidden_sizes=self.hidden_sizes,
                    hidden_activation=self.hidden_activation,
                    bayesian=self.bayesian,
                    batch_size=self.batch_size,
                )
            else:
                self._core_model = MLP(
                    loaded_mlp_model=keras.models.load_model(
                        h5_addr,
                    ),
                    feature_names=self.x_dim,
                    hidden_sizes=self.hidden_sizes,
                    hidden_activation=self.hidden_activation,
                    bayesian=self.bayesian,
                    batch_size=self.batch_size,
                )

        else:
            # create MLP model
            self._core_model = MLP(
                name="mlp_keras_model",
                feature_names=self.x_dim,
                hidden_sizes=self.hidden_sizes,
                hidden_activation=self.hidden_activation,
                bayesian=self.bayesian,
                batch_size=self.batch_size,
                output_layer_config=self.params_config,
                dtype=self.dtype,
            )

    def prob_single(
        self, x: npt.NDArray[np.float64], y: np.float64
    ) -> Tuple[np.float64, np.float64, np.float64]:

        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        # y : np.float64 number
        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [np.expand_dims(x, axis=0), np.expand_dims(y, axis=0)],
        )
        return (
            np.squeeze(pdf.numpy()),
            np.squeeze(log_pdf.numpy()),
            np.squeeze(ecdf.numpy()),
        )

    def prob_batch(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        batch_size=None,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ):
        # to resolve the weird issue when the number of entries are 32*k+1
        single_entry = False
        if len(y) % 32 == 1:
            single_entry = True
            x = np.append(x, [np.ones(len(self.x_dim), dtype=np.float64)], axis=0)
            y = np.append(y, [1.00], axis=0)

        batch_input = [
            tf.expand_dims(tf.convert_to_tensor(x[:, idx], dtype=self.dtype), axis=1)
            for idx in range(len(self.x_dim))
        ]

        batch_input.append(tf.convert_to_tensor(y, dtype=self.dtype))

        # for large batches of input x
        # x : np.array of np.float64 with the shape (?,ndim)
        # y : np.array of np.float64 with the shape (batch_size,1) e.g. np.array([5,6,7,8,9,10])
        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            tuple(batch_input),
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )

        if single_entry:
            return np.squeeze(pdf)[:-1], np.squeeze(log_pdf)[:-1], np.squeeze(ecdf)[:-1]
        else:
            return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def sample_n(
        self,
        x: npt.NDArray[np.float64],
        random_generator: np.random.Generator = np.random.default_rng(),
        batch_size=None,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ) -> npt.NDArray[np.float64]:

        # generate n random numbers uniformly distributed on [0,1]
        y = random_generator.uniform(0, 1, len(x))

        samples = self.sample_model.predict(
            [x, y],
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return np.squeeze(samples)

    def get_parameters(
        self,
        x: npt.NDArray[np.float64],
    ) -> dict:

        # for single value x (not batch)
        # y : np.float64 number
        # for single value x (not batch)
        # x : np.array of np.float64 with the shape (ndim)
        # prediction_res = self.params_model.predict(np.expand_dims(x, axis=0))
        prediction_res = self.params_model.predict(
            tuple([tf.convert_to_tensor([value]) for value in x]),
        )

        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        return result_dict

    def fit(
        self,
        X,
        Y,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.training_model.compile(optimizer=optimizer, loss=self.loss)

        # history = self.core_model.model.fit(
        self.training_model.fit(
            x=[X, Y],
            y=Y,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            # validation_data=(x_val, y_val),
        )

    def fit_pipeline(
        self,
        train_dataset,
        test_dataset,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.pl_training_model.compile(optimizer=optimizer, loss=self.loss)

        # history = self.core_model.model.fit(
        self.pl_training_model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=test_dataset,
            # metrics=[keras.metrics.KLDivergence()]
        )

    @property
    def x_dim(self):
        return self._x_dim

    @property
    def hidden_sizes(self):
        return self._hidden_sizes

    @property
    def hidden_activation(self):
        return self._hidden_activation

class ConditionalRecurrentDensityEstimator(DensityEstimator):

    def __init__(
        self,
        h5_addr: str = None,
        x_dim: int = None,
        recurrent_taps: int = None,
        batch_size: int = None,
        dtype: str = None,
        hidden_layers_config = {},
    ):
        
        super(ConditionalRecurrentDensityEstimator, self).__init__(
            h5_addr=h5_addr,
            batch_size=batch_size,
            dtype=dtype,
        )

        self._x_dim = x_dim
        self._recurrent_taps = recurrent_taps
        self._hidden_layers_config = hidden_layers_config

    def create_core(self, h5_addr: str):

        # initiate the slp model
        if h5_addr is not None:
            # load the keras model and feed to SLP
            self._core_model = RnnMLP(
                loaded_mlp_model=keras.models.load_model(
                    h5_addr,
                ),
                batch_size=self.batch_size,
                recurrent_taps=self.recurrent_taps,
            )
        else:
            # create SLP model
            self._core_model = RnnMLP(
                name="rnnmlp_keras_model",
                feature_names=["target_y",*self._x_dim],
                batch_size=self.batch_size,
                recurrent_taps=self.recurrent_taps,
                hidden_layers_config=self._hidden_layers_config,
                output_layer_config=self.params_config,
                dtype=self.dtype,
            )

    def prob_single(
        self, 
        Y: npt.NDArray[np.float64], 
        X: npt.NDArray[np.float64],
        y: np.float64,
    ) -> Tuple[np.float64, np.float64, np.float64]:
        
        # for single value y (not batch)
        # y : np.float64 number
        # Y : np array with length equal to self.recurrent_taps
        # X : np array with length equal to self.recurrent_taps
        if len(Y) != self.recurrent_taps or len(X) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        Y = np.expand_dims(Y, axis=0)
        X = np.expand_dims(X, axis=0)
        y = np.expand_dims(y, axis=0)
        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [Y, X, y],
            batch_size=1,
        )
        return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def prob_batch(
        self,
        Y: npt.NDArray[np.float64],
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        batch_size=32,
        verbose=0,
        steps=None,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:

        # for large batches of input y
        # y : np.array of np.float64 with the shape (batch_size,1) e.g. np.array([5,6,7,8,9,10])
        # Y : 2D np array with the shape (batch_size, self.recurrent_taps)

        # IMPORTANT: batch size by default in keras is set to 32, if data length is 32*k+1, it raises error.
        if len(y) > batch_size:
            if len(y) % batch_size == 1:
                batch_size = batch_size * 2

        [pdf, log_pdf, ecdf] = self.prob_pred_model.predict(
            [Y, X, y],
            batch_size=batch_size,
            verbose=verbose,
            steps=steps,
            callbacks=None,
            max_queue_size=max_queue_size,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
        )
        return np.squeeze(pdf), np.squeeze(log_pdf), np.squeeze(ecdf)

    def get_parameters(self, Y: npt.NDArray[np.float64], X: npt.NDArray[np.float64]) -> dict:

        # for a single sequence (not batch)
        # Y : np array with length equal to self.recurrent_taps
        if len(Y) != self.recurrent_taps or len(X) != self.recurrent_taps:
            raise Exception("sequence length is not equal to recurrent_taps of the model")

        prediction_res = self.params_model.predict(
            [np.expand_dims(Y, axis=0), np.expand_dims(X, axis=0)]
        )

        result_dict = {}
        for idx, param in enumerate(self.params_config):
            result_dict[param] = np.squeeze(prediction_res[idx])

        return result_dict

    def fit(
        self,
        Y,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.training_model.compile(optimizer=optimizer, loss=self.loss)

        X = np.zeros(len(Y))
        # history = self.core_model.model.fit(
        self.training_model.fit(
            x=[X, Y],
            y=Y,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            # validation_data=(x_val, y_val),
        )

    def fit_pipeline(
        self,
        train_dataset,
        test_dataset,
        optimizer,
        batch_size: int = 1000,
        epochs: int = 10,
    ):

        # this keras model is the one that we use for training
        # self.core_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.pl_training_model.compile(optimizer=optimizer, loss=self.loss)

        # In this train_dataset, there must be an all zero column

        # history = self.core_model.model.fit(
        self.pl_training_model.fit(
            train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            # We pass some validation for
            # monitoring validation loss and metrics
            # at the end of each epoch
            validation_data=test_dataset,
            # metrics=[keras.metrics.KLDivergence()]
        )

    @property
    def recurrent_taps(self):
        return self._recurrent_taps
