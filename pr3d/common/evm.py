import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def safe_log(x, eps=1e-40):
    """log clipped to avoid -inf."""
    return tf.math.log(tf.maximum(x, tf.constant(eps, dtype=x.dtype)))

# ------------------------------------------------------------------
# Helper: safe support check + ξ≈0 (exponential) fallback
# ------------------------------------------------------------------
def _gpd_core(z, xi, scale, norm, dtype):
    """
    Core of the GPD pdf / survival / quantile with full ξ∈ℝ support.
    z        : (y - u) / σ    (non-negative tensor)
    xi       : shape ξ        (can be negative, zero, or positive)
    scale    : σ              (must be >0)
    norm     : “normalising” factor coming from the bulk / tail split
    """
    eps = tf.constant(1e-12, dtype=dtype)
    is_exp = tf.abs(xi) < eps          # ξ≈0 → exponential limit

    # 1 + ξ z  (needed many times)
    one_plus = 1. + xi * z

    # ----------------------------------------------------------------
    # pdf
    # ----------------------------------------------------------------
    pdf_gp  = norm / scale * tf.pow(one_plus, -1. - 1./xi)           # ξ≠0
    pdf_exp = norm / scale * tf.exp(-z)                              # ξ≈0
    pdf     = tf.where(is_exp, pdf_exp, pdf_gp)

    # ----------------------------------------------------------------
    # tail (survival) probability
    # ----------------------------------------------------------------
    sf_gp   = norm * tf.pow(one_plus, -1./xi)                        # ξ≠0
    sf_exp  = norm * tf.exp(-z)                                      # ξ≈0
    sf      = tf.where(is_exp, sf_exp, sf_gp)

    return pdf, sf, one_plus, is_exp


# ------------------------------------------------------------------
# 1)  pdf  (works for any ξ)
# ------------------------------------------------------------------
def gpd_prob(tail_threshold, tail_param, tail_scale, norm_factor,
             y_input, dtype=tf.float64):
    y   = tf.squeeze(y_input)
    u   = tf.squeeze(tail_threshold)
    xi  = tf.squeeze(tail_param)
    σ   = tf.squeeze(tail_scale)
    nrm = tf.squeeze(norm_factor)

    # distance above threshold
    z = tf.maximum(y - u, 0.) / σ

    pdf, _, one_plus, _ = _gpd_core(z, xi, σ, nrm, dtype)

    # enforce support for ξ<0 (upper end-point: u − σ/ξ)
    valid = one_plus > 0.
    return tf.where(valid, pdf, tf.zeros_like(pdf))


# ------------------------------------------------------------------
# 2)  tail (survival) probability  P(Y ≥ y)
# ------------------------------------------------------------------
def gpd_tail_prob(tail_threshold, tail_param, tail_scale, norm_factor,
                  y_input, dtype=tf.float64):
    y   = tf.squeeze(y_input)
    u   = tf.squeeze(tail_threshold)
    xi  = tf.squeeze(tail_param)
    σ   = tf.squeeze(tail_scale)
    nrm = tf.squeeze(norm_factor)

    z = tf.maximum(y - u, 0.) / σ
    _, sf, one_plus, _ = _gpd_core(z, xi, σ, nrm, dtype)

    valid = one_plus > 0.
    return tf.where(valid, sf, tf.zeros_like(sf))


# ------------------------------------------------------------------
# 3)  quantile function  (random_input ~ U(0,1))
# ------------------------------------------------------------------
def gpd_quantile(tail_threshold, tail_param, tail_scale, norm_factor,
                 random_input, dtype=tf.float64):
    u       = tf.squeeze(tail_threshold)
    xi      = tf.squeeze(tail_param)
    σ       = tf.squeeze(tail_scale)
    nrm     = tf.squeeze(norm_factor)
    p       = tf.squeeze(random_input)          # assume already U(0,1)

    eps     = tf.constant(1e-12, dtype=dtype)
    is_exp  = tf.abs(xi) < eps

    # exponential limit
    q_exp = u - σ * safe_log(1. - p / nrm)

    # general ξ≠0 case
    q_gp = u + σ / xi * (tf.pow(1. - p / nrm, -xi) - 1.)

    q = tf.where(is_exp, q_exp, q_gp)

    # clamp to upper end-point when ξ<0  (u − σ/ξ)
    upper = u - σ / tf.where(xi < 0., xi, tf.ones_like(xi))
    q = tf.where((xi < 0.) & (q > upper), upper, q)

    return q

def gpd_log_prob(
    tail_threshold,
    tail_param,
    tail_scale,
    norm_factor,
    y_input,
    dtype=tf.float64,
):
    """
    tensor-based gpd log probability calculation
    """
    # all values are greater than tail_threshold
    return safe_log(
        gpd_prob(
            tail_threshold=tail_threshold,
            tail_param=tail_param,
            tail_scale=tail_scale,
            norm_factor=norm_factor,
            y_input=y_input,
            dtype=dtype,
        ),
    )


def split_bulk_gpd(
    tail_threshold,
    y_input,
    y_batch_size,
    dtype: tf.DType = tf.float64,
):
    # squeez variables
    tail_threshold = tf.squeeze(tail_threshold)
    y_input = tf.squeeze(y_input)

    # gives a tensor, indicating which y_input are greater than tail_threshold
    # greater than threshold is true, else false
    bool_split_tensor = tf.greater(y_input, tail_threshold)  # this is in Boolean

    # find the number of samples in each group
    float_split_tensor = tf.cast(
        bool_split_tensor, dtype=dtype
    )  # convert it to float for multiplication
    tail_samples_count = tf.reduce_sum(float_split_tensor)
    bulk_samples_count = y_batch_size - tail_samples_count

    return bool_split_tensor, tail_samples_count, bulk_samples_count



def split_bulk_gpd_cdf(
    norm_factor,
    random_input,
    dtype: tf.DType = tf.float64,
):

    # squeez variables
    norm_factor = tf.squeeze(norm_factor)
    random_input = tf.squeeze(random_input)

    # gives a tensor, indicating which random_input are greater than norm_factor
    # greater than threshold is true, else false
    bool_split_tensor = tf.greater(random_input, norm_factor)  # this is in Boolean

    return bool_split_tensor


def mixture_tail_prob(
    bool_split_tensor,
    gpd_tail_prob_t,
    bulk_tail_prob_t,
    dtype: tf.DType,
):
    gpd_multiplexer = bool_split_tensor
    bulk_multiplexer = tf.logical_not(bool_split_tensor)

    gpd_multiplexer = tf.cast(
        gpd_multiplexer, dtype=dtype
    )  # convert it to float for multiplication
    bulk_multiplexer = tf.cast(
        bulk_multiplexer, dtype=dtype
    )  # convert it to float for multiplication

    multiplexed_gpd_tail_prob = tf.multiply(gpd_tail_prob_t, gpd_multiplexer)
    multiplexed_bulk_tail_prob = tf.multiply(bulk_tail_prob_t, bulk_multiplexer)

    return tf.add(
        multiplexed_gpd_tail_prob,
        multiplexed_bulk_tail_prob,
    )


def mixture_prob(
    bool_split_tensor,
    gpd_prob_t,
    bulk_prob_t,
    dtype: tf.DType,
):
    gpd_multiplexer = bool_split_tensor
    bulk_multiplexer = tf.logical_not(bool_split_tensor)

    gpd_multiplexer = tf.cast(
        gpd_multiplexer, dtype=dtype
    )  # convert it to float for multiplication
    bulk_multiplexer = tf.cast(
        bulk_multiplexer, dtype=dtype
    )  # convert it to float for multiplication

    multiplexed_gpd_prob = tf.multiply(gpd_prob_t, gpd_multiplexer)
    multiplexed_bulk_prob = tf.multiply(bulk_prob_t, bulk_multiplexer)

    return tf.reduce_sum(
        tf.stack(
            [
                multiplexed_gpd_prob,
                multiplexed_bulk_prob,
            ]
        ),
        axis=0,
    )


def mixture_log_prob(
    bool_split_tensor,
    gpd_prob_t,
    bulk_prob_t,
    dtype: tf.DType,
):

    return safe_log(
        mixture_prob(
            bool_split_tensor=bool_split_tensor,
            gpd_prob_t=gpd_prob_t,
            bulk_prob_t=bulk_prob_t,
            dtype=dtype,
        ),
    )


def mixture_sample(
    cdf_bool_split_t,
    gpd_sample_t,
    bulk_sample_t,
    dtype: tf.DType,
):
    gpd_multiplexer = cdf_bool_split_t
    bulk_multiplexer = tf.logical_not(cdf_bool_split_t)

    gpd_multiplexer = tf.cast(
        gpd_multiplexer, dtype=dtype
    )  # convert it to float (1.00 or 0.00) for multiplication
    bulk_multiplexer = tf.cast(
        bulk_multiplexer, dtype=dtype
    )  # convert it to float (1.00 or 0.00) for multiplication

    multiplexed_gpd_sample = tf.multiply(gpd_sample_t, gpd_multiplexer)
    multiplexed_bulk_sample = tf.multiply(bulk_sample_t, bulk_multiplexer)

    return tf.reduce_sum(
        tf.stack(
            [
                multiplexed_gpd_sample,
                multiplexed_bulk_sample,
            ]
        ),
        axis=0,
    )

