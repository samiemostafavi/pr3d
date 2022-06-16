import numpy as np
import numpy.typing as npt
from typing import Tuple
import keras
import tensorflow as tf
from pr3d.common.tf import SLP
from pr3d.common.bayesian import SavableDenseFlipout

from pr3d.common.tf import SLP

class NonConditionalDensityEstimator(): 
    def __init__(
        self,
        h5_addr : str = None,
        bayesian : bool = False,
        batch_size : int = None,
        dtype : str = 'float64',
    ):
        self._bayesian = bayesian
        self._batch_size = batch_size

        # configure keras to use dtype
        tf.keras.backend.set_floatx(dtype)

        # for creating the tensors
        if dtype == 'float64':
            self._dtype = tf.float64
        elif dtype == 'float32':
            self._dtype = tf.float32
        elif dtype == 'float16':
            self._dtype = tf.float16
        else:
            raise Exception("unknown dtype format")

    def save(self, h5_addr : str):
        pass

    def create_models():
        pass

    def create_slp(self, h5_addr : str):

        # initiate the slp model
        if h5_addr is not None:
            # load the keras model and feed to SLP
            if self.bayesian:
                self._slp_model = SLP(
                    loaded_slp_model = keras.models.load_model(
                        h5_addr, 
                        custom_objects={ 
                            'SavableDenseFlipout' : SavableDenseFlipout,
                        },
                    ),
                    bayesian = self.bayesian,
                    batch_size = self.batch_size,
                )
            else:
                self._slp_model = SLP(
                    loaded_slp_model = keras.models.load_model(
                        h5_addr,
                    ),
                    bayesian = self.bayesian,
                    batch_size = self.batch_size,
                )

        else:
            # create SLP model
            self._slp_model = SLP(
                name = 'slp_keras_model',
                bayesian = self.bayesian,
                batch_size = self.batch_size,
                layer_config=self.params_config,
                dtype = self.dtype,
            )

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
        optimizer,
        batch_size : int = 1000,
        epochs : int = 10,
    ):

        # this keras model is the one that we use for training
        #self.slp_model.model.compile(optimizer=optimizer, loss=self.loss)
        self.training_model.compile(optimizer=optimizer, loss=self.loss)

        X = np.zeros(len(Y))
        #history = self.slp_model.model.fit(
        self.training_model.fit(
            x = [X,Y],
            y = Y,
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
    def training_model(self) -> keras.Model:
        return self._training_model

    @property
    def params_config(self) -> dict:
        return self._params_config

    @property
    def slp_model(self) -> SLP:
        return self._slp_model

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