import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def gmm_nll_loss(centers,dtype):
    # Gaussian mixture network negative log likelihood loss
    def loss(y_true, y_pred):
        # y_pred is the concatenated mixture_weights, mixture_locations, and mixture_scales
        weights = y_pred[:,0:centers-1]
        locs = y_pred[:,centers:2*centers-1]
        scales = y_pred[:,2*centers:3*centers-1]

        # very important line, was causing (batch_size,batch_size)
        y_true = tf.squeeze(y_true)
        
        cat = tfd.Categorical(probs=weights,dtype=dtype)
        components = [tfd.Normal(loc=loc, scale=scale) for loc, scale
                        in zip(tf.unstack(locs, axis=1), tf.unstack(scales, axis=1))]

        mixture = tfd.Mixture(cat=cat, components=components)

        # define logpdf and loglikelihood
        log_pdf_ = mixture.log_prob(y_true)
        log_likelihood_ = tf.reduce_sum(log_pdf_ )
        return -log_likelihood_

    return loss