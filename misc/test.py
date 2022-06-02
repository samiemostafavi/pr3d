import tensorflow as tf
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.internal import prefer_static as ps

def sample_n(_gamma_shape,_gamma_rate,_tail_param,_tail_threshold,_tail_scale, n, dtype, seed):
    seed = samplers.sanitize_seed(seed, salt='gammaEVM')

    gamma_shape = tf.convert_to_tensor(_gamma_shape)
    gamma_rate = tf.convert_to_tensor(_gamma_rate)
    tail_param = tf.convert_to_tensor(_tail_param)
    tail_threshold = tf.convert_to_tensor(_tail_threshold)
    tail_scale = tf.convert_to_tensor(_tail_scale)

    shape=ps.convert_to_shape_tensor([n])
    shape = ps.convert_to_shape_tensor(shape, dtype_hint=tf.int32, name='shape')

    #shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)],
    #                axis=0)
    #sampled = samplers.normal(
    #    shape=shape, mean=0., stddev=1., dtype=self.dtype, seed=seed)
    #return sampled * scale + loc


    # making conditional distribution
    # gamma + gpd

    shape = ps.concat([[n], self._batch_shape_tensor(loc=loc, scale=scale)],
                      axis=0)

    sample = samplers.uniform(
        shape=shape,
        minval=0,
        maxval=1,
        dtype=dtype,
        seed=None,
    )

    # threshold = 0.9;

    if sample < threshold:
        # gamma
        # gamma_scale = 1/2;
        # gamma_shape = 10;
        result = gaminv(sample,gamma_shape,gamma_rate);
    else:
        # gpd
        # gpd_k = 0.5;
        # gpd_sigma = 0.5;
        theta = gaminv(threshold,gamma_shape,gamma_rate);
        gpd_sigma = 1/gampdf(theta,gamma_shape,gamma_rate)*((1-threshold)^2);

        gpd_sample = (sample-threshold)/(1-threshold);
        result = gpinv(gpd_sample,gpd_k,gpd_sigma)/(1-threshold)+theta; 



dtype = tf.float32