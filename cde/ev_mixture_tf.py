import tensorflow as tf

def gpd_prob(
    tail_threshold, 
    tail_param, 
    tail_scale,
    norm_factor,
    y_input, 
    dtype=tf.float64,
):
    """ 
    tensor-based gpd probability calculation
    """
    # y_input could be in batch shape
    # all y_input values are greater than tail_threshold

    # squeeze the variables
    y_input = tf.squeeze(y_input)
    tail_threshold = tf.squeeze(tail_threshold)
    tail_param = tf.squeeze(tail_param)
    tail_scale = tf.squeeze(tail_scale)
    
    # if tail_scale is not zero
    prob = tf.multiply(
        norm_factor,
        tf.multiply(
            tf.divide(
                tf.constant(1.00,dtype=dtype),
                tail_scale,
            ),
            tf.pow(
                tf.add(
                    tf.constant(1.00,dtype=dtype),
                    tf.multiply(
                        tail_param,
                        tf.divide(
                            tf.abs(y_input-tail_threshold),
                            tail_scale,
                        ),
                    ),
                ),
                tf.divide(
                    -tail_param-tf.constant(1.00,dtype=dtype),
                    tail_param,
                ),
            ),
        ),
    )

    return tf.squeeze(prob)

def gpd_tail_prob(
    tail_threshold, 
    tail_param, 
    tail_scale,
    norm_factor,
    y_input,  
    dtype=tf.float64,
):
    """ 
    tensor-based gpd tail probability calculation
    """
    # y_input could be in batch shape
    # all y_input values are greater than tail_threshold

    # squeeze the variables
    y_input = tf.squeeze(y_input)
    tail_threshold = tf.squeeze(tail_threshold)
    tail_param = tf.squeeze(tail_param)
    tail_scale = tf.squeeze(tail_scale)

    # if tail_scale is not zero
    tail_prob = tf.multiply(
        norm_factor,
        tf.pow(
            tf.add(
                tf.constant(1.00,dtype=tf.float64),
                tf.multiply(
                    tf.abs(y_input-tail_threshold),
                    tf.divide(tail_param,tail_scale),
                ),
            ),
            tf.divide(
                tf.constant(-1.00,dtype=tf.float64),
                tail_param,
            ),
        )
    )

    return tf.squeeze(tail_prob)

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
    return tf.log(
        gpd_prob(
            tail_threshold = tail_threshold,
            tail_param = tail_param, 
            tail_scale = tail_scale,
            norm_factor = norm_factor,
            y_input = y_input, 
            dtype = dtype,
        ),
    )

def split_bulk_gpd(
    tail_threshold,
    y_input,
    y_batch_size,
    dtype : tf.DType = tf.float64,
):
    # squeez variables
    tail_threshold = tf.squeeze(tail_threshold)
    y_input = tf.squeeze(y_input)
    
    # gives a tensor, indicating which y_input are greater than tail_threshold
    # greater than threshold is true, else false
    bool_split_tensor = tf.greater(y_input, tail_threshold) # this is in Boolean

    # find the number of samples in each group
    float_split_tensor = tf.cast(bool_split_tensor, dtype=dtype) # convert it to float for multiplication
    tail_samples_count = tf.reduce_sum(float_split_tensor)
    bulk_samples_count = y_batch_size - tail_samples_count

    return bool_split_tensor, tail_samples_count, bulk_samples_count


def mixture_tail_prob(
    bool_split_tensor,
    gpd_tail_prob_t,
    bulk_tail_prob_t,
    dtype: tf.DType,
):
    gpd_multiplexer = bool_split_tensor
    bulk_multiplexer = tf.logical_not(bool_split_tensor)

    gpd_multiplexer = tf.cast(gpd_multiplexer, dtype=dtype) # convert it to float for multiplication
    bulk_multiplexer = tf.cast(bulk_multiplexer, dtype=dtype) # convert it to float for multiplication

    multiplexed_gpd_tail_prob = tf.multiply(gpd_tail_prob_t,gpd_multiplexer)
    multiplexed_bulk_tail_prob = tf.multiply(bulk_tail_prob_t,bulk_multiplexer)

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

    gpd_multiplexer = tf.cast(gpd_multiplexer, dtype=dtype) # convert it to float for multiplication
    bulk_multiplexer = tf.cast(bulk_multiplexer, dtype=dtype) # convert it to float for multiplication

    multiplexed_gpd_prob = tf.multiply(gpd_prob_t,gpd_multiplexer)
    multiplexed_bulk_prob = tf.multiply(bulk_prob_t,bulk_multiplexer)

    return tf.reduce_sum(
        tf.stack([
            multiplexed_gpd_prob,
            multiplexed_bulk_prob,
        ]),
        axis=0,
    )

def mixture_log_prob(
    bool_split_tensor,
    gpd_prob_t,
    bulk_prob_t,
    dtype: tf.DType,
):

    return tf.math.log(
        mixture_prob(
            bool_split_tensor=bool_split_tensor,
            gpd_prob_t=gpd_prob_t,
            bulk_prob_t=bulk_prob_t,
            dtype=dtype,
        ),
    )