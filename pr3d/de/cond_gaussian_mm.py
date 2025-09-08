import h5py
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from keras import layers

from pr3d.common.core import ConditionalDensityEstimator

tfd = tfp.distributions


class ConditionalGaussianMM(ConditionalDensityEstimator):
    def __init__(
        self,
        centers: int = 8,
        x_dim: list | None = None,
        h5_addr: str | None = None,
        bayesian: bool = False,
        batch_size: int | None = None,
        dtype: str = "float64",
        hidden_sizes=(16, 16),
        hidden_activation="tanh",
    ):
        super().__init__(
            x_dim=x_dim,
            h5_addr=h5_addr,
            bayesian=bayesian,
            batch_size=batch_size,
            dtype=dtype,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )

        # --- restore / init hyperparams ---
        if h5_addr is not None:
            with h5py.File(h5_addr, "r") as hf:
                raw = hf.get("x_dim")
                self._x_dim = [s.decode("utf-8") for s in raw[...]] if raw is not None else x_dim
                self._centers = int(hf["centers"][0])
                self._bayesian = bool(hf["bayesian"][0])
                if "batch_size" in hf: self._batch_size = int(hf["batch_size"][0])
                if "hidden_sizes" in hf: self._hidden_sizes = tuple(hf["hidden_sizes"][...].tolist())
                if "hidden_activation" in hf: self._hidden_activation = hf["hidden_activation"][...].astype("S").tobytes().decode()
        else:
            self._x_dim = x_dim
            self._centers = centers
            self._bayesian = bayesian
            self._batch_size = batch_size
            self._hidden_sizes = hidden_sizes
            self._hidden_activation = hidden_activation

        # --- head spec (per-slice) ---
        self._params_config = {
            "mixture_weights": {"slice_size": self.centers, "slice_activation": "softmax"},
            "mixture_locations": {"slice_size": self.centers, "slice_activation": None},
            "mixture_scales": {"slice_size": self.centers, "slice_activation": "softplus"},
        }

        # build the conditional core (MLP with named inputs)
        self.create_core(h5_addr=h5_addr)
        # build inference/training graphs
        self.create_models()

    # ---------- I/O ----------
    def save(self, h5_addr: str) -> None:
        self.core_model.model.save(h5_addr)
        with h5py.File(h5_addr, "a") as hf:
            if "x_dim" in hf: del hf["x_dim"]
            hf.create_dataset("x_dim", data=np.array(self.x_dim, dtype="S"))
            hf.create_dataset("centers", data=np.array([int(self.centers)]))
            hf.create_dataset("bayesian", data=np.array([int(self.bayesian)]))
            if self.batch_size is not None:
                hf.create_dataset("batch_size", data=np.array([int(self.batch_size)]))
            if self.hidden_sizes is not None:
                hf.create_dataset("hidden_sizes", data=np.array(self.hidden_sizes, dtype=np.int32))
            if self.hidden_activation is not None:
                hf.create_dataset("hidden_activation", data=np.array([str(self.hidden_activation)], dtype="S"))

    # ---------- graphs ----------
    def create_models(self):
        # inputs from the conditional core
        # dict[str, KerasTensor] with stable names matching feature names
        x_inputs_dict = self.core_model.input_slices
        x_inputs_list = list(x_inputs_dict.values())  # for convenience if needed

        # heads
        self.weights = self.core_model.output_slices["mixture_weights"]   # (None, K)
        self.locs    = self.core_model.output_slices["mixture_locations"] # (None, K)
        self.scales  = self.core_model.output_slices["mixture_scales"]    # (None, K)

        # params model (dict inputs → 3 heads)
        self._params_model = keras.Model(
            inputs=x_inputs_dict,
            outputs=[self.weights, self.locs, self.scales],
            name="params_model",
        )

        # target input
        self.y_input = keras.Input(
            name="y_input",
            shape=(1,),
            batch_size=self.batch_size,
            dtype=self.dtype,
        )
        y_flat = layers.Lambda(lambda y: tf.squeeze(y, axis=-1), name="y_flat")(self.y_input)  # (None,)

        # TFP mixture via MixtureSameFamily inside Lambda (Keras-safe)
        def _pdf(args):
            w, m, s, y = args
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=w),
                components_distribution=tfd.Normal(loc=m, scale=s),
            )
            return mix.prob(y)

        def _logpdf(args):
            w, m, s, y = args
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=w),
                components_distribution=tfd.Normal(loc=m, scale=s),
            )
            return mix.log_prob(y)

        def _cdf(args):
            w, m, s, y = args
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=w),
                components_distribution=tfd.Normal(loc=m, scale=s),
            )
            return mix.cdf(y)

        self.pdf     = layers.Lambda(_pdf,    name="pdf",     output_shape=(None,))([self.weights, self.locs, self.scales, y_flat])
        self.log_pdf = layers.Lambda(_logpdf, name="log_pdf", output_shape=(None,))([self.weights, self.locs, self.scales, y_flat])
        self.ecdf    = layers.Lambda(_cdf,    name="ecdf",    output_shape=(None,))([self.weights, self.locs, self.scales, y_flat])

        # training output expects shape (None,1)
        self.expanded_log_pdf = layers.Lambda(lambda z: tf.expand_dims(z, -1), name="expanded_log_pdf")(self.log_pdf)

        # prob-prediction model (dict inputs + y)
        self._prob_pred_model = keras.Model(
            inputs={**x_inputs_dict, "y_input": self.y_input},
            outputs=[self.pdf, self.log_pdf, layers.Lambda(lambda z: tf.expand_dims(z, -1), name="ecdf_exp")(self.ecdf)],
            name="prob_pred_model",
        )

        # pipeline / training models (dict inputs)
        self._pl_training_model = keras.Model(
            inputs={**x_inputs_dict, "y_input": self.y_input},
            outputs=[self.expanded_log_pdf],
            name="pl_training_model",
        )
        self._training_model = keras.Model(
            inputs={**x_inputs_dict, "y_input": self.y_input},
            outputs=[self.expanded_log_pdf],
            name="training_model",
        )

        # mean-NLL (more comparable across batch sizes)
        self._loss = lambda y_true, y_pred: -tf.reduce_mean(y_pred)

    # ---------- convenience ----------
    @property
    def centers(self): return self._centers

    def mean(self, x: dict[str, npt.NDArray[np.float64]]):
        """Mixture mean E[Y|X=x] via params_model."""
        weights, locs, scales = self._params_model.predict(x, verbose=0)
        mix = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=tf.convert_to_tensor(weights, dtype=self.dtype)),
            components_distribution=tfd.Normal(
                loc=tf.convert_to_tensor(locs, dtype=self.dtype),
                scale=tf.convert_to_tensor(scales, dtype=self.dtype),
            ),
        )
        return mix.mean().numpy()

    def quantile(
        self,
        x: dict[str, npt.NDArray[np.float64]],
        samples: npt.NDArray[np.float64],   # probabilities in (0,1)
        value_tolerance=1e-7,
        position_tolerance=1e-3,
    ):
        """Vectorized numerical quantile finder for mixture CDF."""
        # initial guess: mixture mean
        x_mean = self.mean(x).astype(np.float64)

        # define objective F(q) - p for each item
        def obj(q):
            # q: Tensor of shape (N,)
            # build CDF(q | x)
            weights, locs, scales = self._params_model.predict(x, verbose=0)
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=tf.convert_to_tensor(weights, dtype=self.dtype)),
                components_distribution=tfd.Normal(
                    loc=tf.convert_to_tensor(locs, dtype=self.dtype),
                    scale=tf.convert_to_tensor(scales, dtype=self.dtype),
                ),
            )
            return mix.cdf(q) - tf.convert_to_tensor(samples, dtype=self.dtype)

        roots = tfp.math.find_root_secant(
            objective_fn=obj,
            initial_position=tf.convert_to_tensor(x_mean.squeeze(), dtype=self.dtype),
            value_tolerance=tf.convert_to_tensor(np.full_like(samples, value_tolerance, dtype=np.float64), dtype=self.dtype),
            position_tolerance=tf.convert_to_tensor(np.full_like(samples, position_tolerance, dtype=np.float64), dtype=self.dtype),
        )
        return roots[0].numpy()

    def sample_n(self, x: dict[str, npt.NDArray[np.float64]], seed: int = 0):
        """Sample from the conditional mixture via component selection + Normal quantile."""
        N = len(next(iter(x.values())))
        weights, locs, scales = self._params_model.predict(x, verbose=0)  # shapes (N,K)
        weights = np.asarray(weights, dtype=np.float64)

        # choose component per sample
        cat_idx = tf.random.categorical(tf.math.log(weights), num_samples=1, seed=seed)
        cat_idx = tf.squeeze(cat_idx, axis=1)  # (N,)

        locs_t   = tf.gather(tf.convert_to_tensor(locs,   dtype=self.dtype), cat_idx, axis=1, batch_dims=1)
        scales_t = tf.gather(tf.convert_to_tensor(scales, dtype=self.dtype), cat_idx, axis=1, batch_dims=1)

        comp = tfd.Normal(loc=locs_t, scale=scales_t)
        u = tf.convert_to_tensor(np.random.uniform(size=N), dtype=self.dtype)
        return comp.quantile(u).numpy()
