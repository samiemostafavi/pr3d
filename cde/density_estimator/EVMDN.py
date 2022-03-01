from random import randint
import numpy as np
import tensorflow as tf
from edward.models import Categorical, Mixture, Normal, MultivariateNormalDiag
from cde.utils.tf_utils.network import MLP
import cde.utils.tf_utils.layers as L
from cde.utils.tf_utils.layers_powered import LayersPowered
from cde.utils.serializable import Serializable
from cde.utils.tf_utils.adamW import AdamWOptimizer
from scipy.integrate import quad
from cde.utils.optimizers import new_find_root_by_bounding

from .BaseNNMixtureEstimator import BaseNNMixtureEstimator

class GPDTailDistribution():
  """ GPD Tail Distribution
      Args:

  """
  def __init__(self, tail_threshold, tail_param, tail_scale):
    self.tail_threshold = tail_threshold
    self.tail_param = tail_param
    self.tail_scale = tail_scale

  # pdf
  def prob(self, values, norm_factor):
    # all values are greater than tail_threshold
    self.values_sq = tf.squeeze(values)
    self.us = tf.squeeze(self.tail_threshold)
    self.zais = tf.squeeze(self.tail_param)
    self.sigmas = tf.squeeze(self.tail_scale)
    # if zais are not zero
    self.tres = tf.multiply(norm_factor,tf.multiply(tf.divide(tf.constant(1.00,dtype=tf.float64),self.sigmas),tf.pow(tf.constant(1.00,dtype=tf.float64)+tf.multiply(self.zais,tf.divide(tf.abs(self.values_sq-self.us),self.sigmas)),tf.divide(-self.zais-tf.constant(1.00,dtype=tf.float64),self.zais))))
    return tf.squeeze(self.tres)
  
  # 1-cdf(y)
  def tail_prob(self, values, norm_factor):
    self.cvalues_sq = tf.squeeze(values)
    self.cus = tf.squeeze(self.tail_threshold)
    self.zais = tf.squeeze(self.tail_param)
    self.sigmas = tf.squeeze(self.tail_scale)
    # if zais are not zero
    self.ctres = tf.multiply(norm_factor,tf.pow(tf.constant(1.00,dtype=tf.float64)+tf.multiply(tf.abs(self.cvalues_sq-self.cus),tf.divide(self.zais,self.sigmas)),tf.divide(tf.constant(-1.00,dtype=tf.float64),self.zais)))
    return tf.squeeze(self.ctres)

  # ln(pdf)
  def log_prob(self, values, norm_factor):
    # all values are greater than tail_threshold
    return tf.log(self.prob(values=values,norm_factor=norm_factor))


class NoNaNGPDExtremeValueMixtureDensityNetwork(BaseNNMixtureEstimator):
  """ GPD Extreme Value Mixture Density Network Estimator

    See "Mixture Density networks", Bishop 1994

    Args:
        name: (str) name space of MDN (should be unique in code, otherwise tensorflow namespace collitions may arise)
        ndim_x: (int) dimensionality of x variable
        ndim_y: (int) dimensionality of y variable
        n_centers: Number of Gaussian mixture components
        hidden_sizes: (tuple of int) sizes of the hidden layers of the neural network
        hidden_nonlinearity: (tf function) nonlinearity of the hidden layers
        n_training_epochs: Number of epochs for training
        x_noise_std: (optional) standard deviation of Gaussian noise over the the training data X -> regularization through noise
        y_noise_std: (optional) standard deviation of Gaussian noise over the the training data Y -> regularization through noise
        adaptive_noise_fn: (callable) that takes the number of samples and the data dimensionality as arguments and returns
                                   the noise std as float - if used, the x_noise_std and y_noise_std have no effect
        entropy_reg_coef: (optional) scalar float coefficient for shannon entropy penalty on the mixture component weight distribution
        weight_decay: (float) the amount of decoupled (http://arxiv.org/abs/1711.05101) weight decay to apply
        l2_reg: (float) the amount of l2 penalty on neural network weights
        l1_reg: (float) the amount of l1 penalty on neural network weights
        weight_normalization: (boolean) whether weight normalization shall be used
        data_normalization: (boolean) whether to normalize the data (X and Y) to exhibit zero-mean and std
        dropout: (float) the probability of switching off nodes during training
        random_seed: (optional) seed (int) of the random number generators used
    """


  def __init__(self, name, ndim_x, ndim_y, n_centers=10, hidden_sizes=(16, 16), hidden_nonlinearity=tf.nn.tanh,
               n_training_epochs=1000, x_noise_std=None, y_noise_std=None, adaptive_noise_fn=None, entropy_reg_coef=0.0,
               weight_decay=0.0, weight_normalization=False, data_normalization=False, dropout=0.0, l2_reg=0.0, l1_reg=0.0,
               random_seed=None,verbose_step=100,learning_rate=5e-3,epsilon=1e-8):

    Serializable.quick_init(self, locals())
    self._check_uniqueness_of_scope(name)

    self.name = name
    self.ndim_x = ndim_x
    self.ndim_y = 1

    self.random_seed = random_seed
    self.random_state = np.random.RandomState(seed=random_seed)
    tf.set_random_seed(random_seed)

    self.n_centers = n_centers

    self.hidden_sizes = hidden_sizes
    self.hidden_nonlinearity = hidden_nonlinearity

    self.n_training_epochs = n_training_epochs

    self.verbose_step = verbose_step
    self.learning_rate = learning_rate
    self.epsilon = epsilon

    # regularization parameters
    self.x_noise_std = x_noise_std
    self.y_noise_std = y_noise_std
    self.entropy_reg_coef = entropy_reg_coef
    self.adaptive_noise_fn = adaptive_noise_fn
    self.weight_decay = weight_decay
    self.l2_reg = l2_reg
    self.l1_reg = l1_reg
    self.weight_normalization = weight_normalization
    self.data_normalization = data_normalization
    self.dropout = dropout

    self.can_sample = True
    self.has_pdf = True
    self.has_cdf = True

    self.fitted = False

    # build tensorflow model
    self._build_model()

  def split_values(self,values):
    us = tf.squeeze(self.tail_threshold)
    values_sq = tf.squeeze(values)
    
    # NEW CHANGE
    greaterboolmat = tf.greater(values_sq,us) 
    #sigslope = tf.constant(10.0, dtype = tf.float32)
    #greaterboolmat = tf.sigmoid(tf.multiply(sigslope,values_sq-us))

    batchsize = tf.cast(tf.size(values),dtype=tf.float64)
    tailsamplescount = tf.reduce_sum(tf.cast(greaterboolmat, dtype=tf.float64))
    bulksamplescount = batchsize - tailsamplescount 
    return greaterboolmat, tailsamplescount, bulksamplescount, batchsize

  def bulk_cumulative_prob(self,value):
    res = tf.zeros_like(tf.squeeze(value))
    self.tres = res
    for loc, scale, weight in zip(tf.unstack(self.locs, axis=1), tf.unstack( self.scales, axis=1), tf.unstack(self.weights , axis=1)):
      self.tloc = tf.squeeze(loc)
      self.tscale = tf.squeeze(scale)
      self.tweight = tf.squeeze(weight)
      dist = tf.distributions.Normal(loc=tf.squeeze(loc), scale=tf.squeeze(scale))
      self.tcdf = dist.cdf(tf.squeeze(value))
      self.tmult = resmult = tf.multiply(dist.cdf(tf.squeeze(value)),tf.squeeze(weight))
      res = res + resmult
    return res
  
  def tail_prob(self,values):
    self.tail_tp_nonmlt = self.tail_dist.tail_prob(values=values,norm_factor=self.norm_factor)
    self.bulk_tp_nonmlt = tf.constant(1.00,dtype=tf.float64)-self.bulk_cumulative_prob(values)
    
    # NEW CHANGE
    self.tailmultiplex = tf.cast(self.greaterboolmat, dtype=tf.float64)
    self.bulkmultiplex = tf.cast(tf.logical_not(self.greaterboolmat), dtype=tf.float64)
    #self.tailmultiplex = self.greaterboolmat
    #self.bulkmultiplex = tf.constant(1.0,dtype=tf.float32)-self.greaterboolmat

    self.tailtp = tf.multiply(self.tail_tp_nonmlt,self.tailmultiplex)
    self.bulktp = tf.multiply(self.bulk_tp_nonmlt,self.bulkmultiplex)
    return self.tailtp+self.bulktp

  def prob(self,values):
    self.tailprobs_nonmlt = self.tail_dist.prob(values=values,norm_factor=self.norm_factor)
    self.bulkprobs_nonmlt = self.mixture.prob(values)
    
    # NEW CHANGE
    self.tailmultiplex = tf.cast(self.greaterboolmat, dtype=tf.float64)
    self.bulkmultiplex = tf.cast(tf.logical_not(self.greaterboolmat), dtype=tf.float64)
    #self.tailmultiplex = self.greaterboolmat
    #self.bulkmultiplex = tf.constant(1.0,dtype=tf.float32)-self.greaterboolmat

    self.tailprobs = tf.multiply(self.tailprobs_nonmlt,self.tailmultiplex)
    self.bulkprobs = tf.multiply(self.bulkprobs_nonmlt,self.bulkmultiplex)
    return self.tailprobs+self.bulkprobs

  def log_prob(self,values):
    return tf.log(self.prob(values))

  def log_like(self, values):
    self.taillogprobs_nonmlt = self.tail_dist.log_prob(values=values,norm_factor=self.norm_factor)
    #self.tail_log_like = self.tail_dist.log_like(values=values,norm_factor=self.norm_factor,k=self.k)
    self.bulklogprobs_nonmlt = self.mixture.log_prob(values)
    
    # NEW CHANGE
    self.tailmultiplex = tf.cast(self.greaterboolmat, dtype=tf.float64)
    self.bulkmultiplex = tf.cast(tf.logical_not(self.greaterboolmat), dtype=tf.float64)
    #self.tailmultiplex = self.greaterboolmat
    #self.bulkmultiplex = tf.constant(1.0,dtype=tf.float32)-self.greaterboolmat

    self.taillogprobs = tf.multiply(self.taillogprobs_nonmlt,self.tailmultiplex)
    self.bulklogprobs = tf.multiply(self.bulklogprobs_nonmlt,self.bulkmultiplex)

    self.tail_log_like = tf.reduce_sum(self.taillogprobs)
    self.bulk_log_like = tf.reduce_sum(self.bulklogprobs)
    
    return self.bulk_log_like + self.tail_log_like

  """ test prob function """
  #mx = np.array([[0,1,1]])
  #my = np.array([[10]])
  #print(model.pdf(mx,my))

  def pdf(self, X, Y):
    assert self.fitted, "model must be fitted to compute probability"
    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
    p = self.sess.run(self.pdf_, feed_dict={self.X_ph: X, self.Y_ph: Y})
    #assert p.ndim == 1 and p.shape[0] == X.shape[0]
    return p

  """ test tail prob """
  #cond_state = [0,1,1]
  #mx = np.array([cond_state])
  #my = np.array([11])
  #print(model.tail(mx,my))
  #my = np.array([9])
  #print(model.tail(mx,my))

  # 1-cdf
  def tail(self, X, Y):
    assert self.fitted, "model must be fitted to compute tail probability"
    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
    p = self.sess.run(self.tail_, feed_dict={self.X_ph: X, self.Y_ph: Y})
    #assert p.ndim == 1 and p.shape[0] == X.shape[0]
    return p

  # (1-cdf(y)) = T -> y?
  def tail_inverse(self, X, T, init_bound, eps):
      assert self.fitted, "model must be fitted to compute tail probability"

      tail_root = lambda y: T - self.tail(X,y)
      init_bound = init_bound * np.ones(X.shape[0])
      return new_find_root_by_bounding(tail_root, left=np.array([0]), right=np.array([init_bound]), eps=eps)[0][0]


  def fit(self, X, Y, random_seed=None, verbose=True, eval_set=None, **kwargs):
    """
    Fit the model with to the provided data

    :param X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
    :param Y: numpy array of y targets - shape: (n_samples, n_dim_y)
    :param eval_set: (tuple) eval/test dataset - tuple (X_test, Y_test)
    :param verbose: (boolean) controls the verbosity of console output
    """  

    X, Y = self._handle_input_dimensionality(X, Y, fitting=True)

    if eval_set is not None:
      eval_set = self._handle_input_dimensionality(*eval_set)

    # If no session has yet been created, create one and make it the default
    self.sess = tf.get_default_session() if tf.get_default_session() else tf.InteractiveSession()

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    tf.initializers.variables(var_list, name='init').run()

    if self.data_normalization:
        self._compute_data_normalization(X, Y)

    #self._compute_noise_intensity(X, Y)

    for i in range(0, self.n_training_epochs + 1):

      self.sess.run(self.train_step,
                  feed_dict={self.X_ph: X, self.Y_ph: Y, self.train_phase: True, self.dropout_ph: self.dropout})

      if verbose and not i % self.verbose_step:
        eval_log_loss = 0
        # compute evaluation loss
        if eval_set is not None:
          X_test, Y_test = eval_set
          eval_log_loss = self.sess.run(self.log_loss_, feed_dict={self.X_ph: X_test, self.Y_ph: Y_test})

        log_loss = self.sess.run(self.log_loss_, feed_dict={self.X_ph: X, self.Y_ph: Y})
        print("Step " + str(i) + ": train log-likelihood " + str(log_loss) + ", eval log-likelihood " + str(eval_log_loss))

    self.fitted = True

  def _build_model(self):
    """
    implementation of the MDN
    """

    with tf.variable_scope(self.name):
      # adds placeholders, data_normalization and data_noise if desired. Also adds a placeholder for dropout probability
      self.layer_in_x, self.layer_in_y = self._build_input_layers()

      # create core multi-layer perceptron
      # Bulk density parameters: 2 * self.ndim_y * self.n_centers + self.n_centers
      # GPD Tail density parameters: 3 (tail param, tail threshold, tail scale)
      mlp_output_dim = 2 * self.ndim_y * self.n_centers + self.n_centers + 3
      core_network = MLP(
              name="core_network",
              input_layer=self.layer_in_x,
              output_dim=mlp_output_dim,
              hidden_sizes=self.hidden_sizes,
              hidden_nonlinearity=self.hidden_nonlinearity,
              output_nonlinearity=None,
              weight_normalization=self.weight_normalization,
              dropout_ph=self.dropout_ph if self.dropout else None
          )

      core_output_layer = core_network.output_layer

      # slice output of MLP into three equally sized parts for loc, scale and mixture weights
      slice_layer_locs = L.SliceLayer(core_output_layer, indices=slice(0, self.ndim_y * self.n_centers), axis=-1)
      slice_layer_scales = L.SliceLayer(core_output_layer, indices=slice(self.ndim_y * self.n_centers, 2 * self.ndim_y * self.n_centers), axis=-1)
      slice_layer_weights = L.SliceLayer(core_output_layer, indices=slice(2 * self.ndim_y * self.n_centers, 2 * self.ndim_y * self.n_centers + self.n_centers), axis=-1)
      slice_layer_tail_param = L.SliceLayer(core_output_layer, indices=slice(2 * self.ndim_y * self.n_centers + self.n_centers, mlp_output_dim-2), axis=-1)
      slice_layer_tail_threshold = L.SliceLayer(core_output_layer, indices=slice(mlp_output_dim-2, mlp_output_dim-1), axis=-1)
      slice_layer_tail_scale = L.SliceLayer(core_output_layer, indices=slice(mlp_output_dim-1, mlp_output_dim), axis=-1)

      # locations mixture components
      self.reshape_layer_locs = L.ReshapeLayer(slice_layer_locs, (-1, self.n_centers, self.ndim_y))
      #self.softplus_layer_locs = L.NonlinearityLayer(self.reshape_layer_locs)
      self.softplus_layer_locs = self.reshape_layer_locs
      self.locs = L.get_output(self.softplus_layer_locs)

      # scales of the mixture components
      reshape_layer_scales = L.ReshapeLayer(slice_layer_scales, (-1, self.n_centers, self.ndim_y))
      self.softplus_layer_scales = L.NonlinearityLayer(reshape_layer_scales, nonlinearity=tf.nn.softplus)
      self.scales = L.get_output(self.softplus_layer_scales)

      # weights of the mixture components
      self.logits = L.get_output(slice_layer_weights)
      self.softmax_layer_weights = L.NonlinearityLayer(slice_layer_weights, nonlinearity=tf.nn.softmax)
      self.weights = L.get_output(self.softmax_layer_weights)

      # tail parameter (zai)
      self.softplus_layer_tail_param = L.NonlinearityLayer(slice_layer_tail_param, nonlinearity=tf.nn.softplus)
      self.tail_param = L.get_output(self.softplus_layer_tail_param)

      # tail threshold (u)
      #self.softplus_tail_threshold = L.NonlinearityLayer(slice_layer_tail_threshold)
      self.softplus_tail_threshold = slice_layer_tail_threshold
      self.tail_threshold = L.get_output(self.softplus_tail_threshold)

      # tail scale (sigma)
      self.softplus_tail_scale = L.NonlinearityLayer(slice_layer_tail_scale, nonlinearity=tf.nn.softplus)
      self.tail_scale = L.get_output(self.softplus_tail_scale)

      # # put mixture components together
      self.y_input = L.get_output(self.layer_in_y)
      self.cat = cat = Categorical(logits=self.logits)
      self.components = components = [MultivariateNormalDiag(loc=loc, scale_diag=scale) for loc, scale
                     in zip(tf.unstack(self.locs, axis=1), tf.unstack( self.scales, axis=1))]
      self.mixture = mixture = Mixture(cat=cat, components=components, value=tf.zeros_like(self.y_input))
      
      # create a power law tail probability estimator
      self.tail_dist = GPDTailDistribution(tail_threshold=self.tail_threshold,tail_param=self.tail_param,tail_scale=self.tail_scale)

      # regularization
      self._add_softmax_entropy_regularization()
      self._add_l1_l2_regularization(core_network)

      self.greaterboolmat, self.tailsamplescount, self.bulksamplescount, self.batchsize = self.split_values(self.y_input)
      #self.norm_factor = 1.00-self.mixture.cdf(self.tail_threshold)
      self.norm_factor = 1.00-self.bulk_cumulative_prob(self.tail_threshold)

      # tensor to compute probabilities
      if self.data_normalization:
        self.pdf_ = self.prob(self.y_input) / tf.reduce_prod(self.std_y_sym)
        self.log_pdf_ = self.log_prob(self.y_input) - tf.reduce_sum(tf.log(self.std_y_sym))
      else:
        self.pdf_ = self.prob(self.y_input)
        self.log_pdf_ = self.log_prob(self.y_input)

      self.tail_ = self.tail_prob(self.y_input)

      # symbolic tensors for getting the unnormalized mixture components
      if self.data_normalization:
        self.scales_unnormalized = self.scales * self.std_y_sym
        self.locs_unnormalized = self.locs * self.std_y_sym + self.mean_y_sym
      else:
        self.scales_unnormalized = self.scales
        self.locs_unnormalized = self.locs

      self.log_like_ = self.log_like(self.y_input)
      self.log_loss_ = -self.log_like_

      optimizer = AdamWOptimizer(self.weight_decay, learning_rate=self.learning_rate, epsilon=self.epsilon) #if self.weight_decay else tf.train.AdamOptimizer()
      self.train_step = optimizer.minimize(self.log_loss_)

    # initialize LayersPowered --> provides functions for serializing tf models
    LayersPowered.__init__(self, [self.softplus_tail_scale, self.softplus_tail_threshold, self.softplus_layer_tail_param, self.softmax_layer_weights, self.softplus_layer_scales, self.softplus_layer_locs,
                                   self.layer_in_y])

  def _param_grid(self):
    param_grid = {
        "n_training_epochs": [500, 1000],
        "n_centers": [5, 10, 20],
        "x_noise_std": [0.1, 0.15, 0.2, 0.3],
        "y_noise_std": [0.1, 0.15, 0.2]
    }
    return param_grid

  def _get_mixture_components(self, X):
    weights, locs, scales = self.sess.run([self.weights, self.locs_unnormalized, self.scales_unnormalized], feed_dict={self.X_ph: X})
    #assert weights.shape[0] == locs.shape[0] == scales.shape[0] == X.shape[0]
    #assert weights.shape[1] == locs.shape[1] == scales.shape[1] == self.n_centers
    #assert locs.shape[2] == scales.shape[2] == self.ndim_y
    #assert locs.ndim == 3 and scales.ndim == 3 and weights.ndim == 2
    return weights, locs, scales

  def _get_tail_components(self, X):
    threshold, tail_param, tail_scale = self.sess.run([self.tail_threshold, self.tail_param, self.tail_scale], feed_dict={self.X_ph: X})
    return threshold, tail_param, tail_scale


  """ verbose function """
  #mx = np.array([[0,1,1],[0,1,1],[0,1,1],[0,1,1],[0,1,1]])
  #my = np.array([[7],[5.2],[6.44],[2],[10]])
  #model._verbose(mx,my)

  def _verbose(self, X, Y):
    X, Y = self._handle_input_dimensionality(X, Y, fitting=False)
    print("X: "+str(X))
    print("Y: "+str(Y))

    print("scales: " + str(np.squeeze(self.sess.run(self.scales,feed_dict={self.X_ph: X, self.Y_ph: Y}))))
    print("weights: " + str(np.squeeze(self.sess.run(self.weights,feed_dict={self.X_ph: X, self.Y_ph: Y}))))
    print("locs: " + str(np.squeeze(self.sess.run(self.locs,feed_dict={self.X_ph: X, self.Y_ph: Y}))))
    print("tail_threshold: " + str(np.squeeze(self.sess.run(self.tail_threshold,feed_dict={self.X_ph: X, self.Y_ph: Y}))))
    print("tail_param: " + str(np.squeeze(self.sess.run(self.tail_param,feed_dict={self.X_ph: X, self.Y_ph: Y}))))
    print("tail_scale: " + str(np.squeeze(self.sess.run(self.tail_scale,feed_dict={self.X_ph: X, self.Y_ph: Y}))))
    print("tailsamplescount: " + str(self.sess.run(self.tailsamplescount,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("bulksamplescount: " + str(self.sess.run(self.bulksamplescount,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("batchsize: " + str(self.sess.run(self.batchsize,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("norm_factor: " + str(self.sess.run(self.norm_factor,feed_dict={self.X_ph: X, self.Y_ph: Y})))

    print("tail values_sq: " + str(self.sess.run(self.tail_dist.values_sq,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("tail us: " + str(self.sess.run(self.tail_dist.us,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("tail zais: " + str(self.sess.run(self.tail_dist.zais,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("tail sigmas: " + str(self.sess.run(self.tail_dist.sigmas,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("tail res:" + str(self.sess.run(self.tail_dist.tres,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    
    print("taillogprobs_nonmlt:" + str(self.sess.run(self.taillogprobs_nonmlt,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("bulklogprobs_nonmlt:" + str(self.sess.run(self.bulklogprobs_nonmlt,feed_dict={self.X_ph: X, self.Y_ph: Y})))

    print("tailmultiplex:" + str(self.sess.run(self.tailmultiplex,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("bulkmultiplex:" + str(self.sess.run(self.bulkmultiplex,feed_dict={self.X_ph: X, self.Y_ph: Y})))

    print("taillogprobs:" + str(self.sess.run(self.taillogprobs,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("bulklogprobs:" + str(self.sess.run(self.bulklogprobs,feed_dict={self.X_ph: X, self.Y_ph: Y})))

    print("tail_log_like:" + str(self.sess.run(self.tail_log_like,feed_dict={self.X_ph: X, self.Y_ph: Y})))
    print("bulk_log_like:" + str(self.sess.run(self.bulk_log_like,feed_dict={self.X_ph: X, self.Y_ph: Y})))

    print("log_loss_:" + str(self.sess.run(self.log_loss_,feed_dict={self.X_ph: X, self.Y_ph: Y})))


  def __str__(self):
    return "\nEstimator type: {}\n n_centers: {}\n entropy_reg_coef: {}\n data_normalization: {} \n weight_normalization: {}\n" \
             "n_training_epochs: {}\n x_noise_std: {}\n y_noise_std: {}\n ".format(self.__class__.__name__, self.n_centers, self.entropy_reg_coef,
                                                  self.data_normalization, self.weight_normalization, self.n_training_epochs, self.x_noise_std, self.y_noise_std)
