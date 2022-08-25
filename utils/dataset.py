import numpy as np
import pyarrow as pa

# import pyarrow.compute as pc
import pyarrow.parquet as pq
import tensorflow as tf
import tensorflow_io as tfio


def parquet_tf_pipeline(
    file_addr,
    feature_names,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float32,
):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = tfio.IODataset.from_parquet(
        filename=file_addr,
    ).prefetch(buffer_size=dataset_size)

    def read_parquet(features):
        # features is an OrderedDict

        # prepare empty tensors
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values_dict = {}

        for a in features.items():
            # look for the features
            for idx, feature_name in enumerate(feature_names):
                # a is a tuple, first item is the key, second is the tensor
                if a[0].decode("utf-8") == feature_name:
                    values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                    values = values.write(values.size(), tf.cast(a[1], dtype=dtype))
                    # important to have the squeeze to get (None,) tensor shape
                    values_dict[feature_name] = tf.squeeze(values.stack(), axis=0)

            # look for the keys
            if a[0].decode("utf-8") == label_name:
                keys = keys.write(keys.size(), tf.cast(a[1], dtype=dtype))

        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(), axis=0))

    dataset = dataset.map(read_parquet)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size)
        .cache()
        .shuffle(buffer_size=train_size)
        .batch(batch_size)
    )
    test_dataset = (
        dataset.skip(train_size)
        .take(dataset_size - train_size)
        .cache()
        .shuffle(buffer_size=train_size)
        .batch(batch_size)
    )

    # to check what is being read:
    # for ds in train_dataset:
    #    print(tfds.as_numpy(ds))
    # for ds in test_dataset:
    #    print(tfds.as_numpy(ds))

    return train_dataset, test_dataset


def parquet_tf_pipeline_2(
    file_addr,
    feature_names,
    label_name,
    dataset_size,
    train_size,
    batch_size,
    dtype=tf.float32,
):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset = tfio.IODataset.from_parquet(
        filename=file_addr,
    ).prefetch(buffer_size=dataset_size)

    def read_parquet(features):
        # features is an OrderedDict

        # prepare empty tensors
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values_dict = {}

        for a in features.items():
            # look for the features
            for idx, feature_name in enumerate(feature_names):
                # a is a tuple, first item is the key, second is the tensor
                if a[0].decode("utf-8") == feature_name:
                    values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                    values = values.write(values.size(), tf.cast(a[1], dtype=dtype))
                    # important to have the squeeze to get (None,) tensor shape
                    values_dict[feature_name] = tf.squeeze(values.stack(), axis=0)

            # look for the keys
            if a[0].decode("utf-8") == label_name:
                keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                keys = keys.write(keys.size(), tf.cast(a[1], dtype=dtype))
                values_dict["y_input"] = tf.squeeze(keys.stack(), axis=0)

        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(), axis=0))

    dataset = dataset.map(read_parquet)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size)
        .cache()
        .shuffle(buffer_size=train_size)
        .batch(batch_size)
    )
    test_dataset = (
        dataset.skip(train_size)
        .take(dataset_size - train_size)
        .cache()
        .shuffle(buffer_size=train_size)
        .batch(batch_size)
    )

    # to check what is being read:
    # for ds in train_dataset:
    #    print(tfds.as_numpy(ds))
    # for ds in test_dataset:
    #    print(tfds.as_numpy(ds))

    return train_dataset, test_dataset


def create_dataset(
    n_samples=300, x_dim=3, x_max=10, x_level=2, dtype="float64", dist="normal"
):

    # generate random sample, two components
    X = np.array(np.random.randint(x_max, size=(n_samples, x_dim)) * x_level).astype(
        dtype
    )

    if dist == "normal":
        Y = np.array(
            [
                np.random.normal(
                    loc=x_sample[0] + x_sample[1] + x_sample[2],
                    scale=(x_sample[0] + x_sample[1] + x_sample[2]) / 5,
                )
                for x_sample in X
            ]
        ).astype(dtype)
    elif dist == "gamma":
        Y = np.array(
            [
                np.random.gamma(
                    shape=x_sample[0] + x_sample[1] + x_sample[2],
                    scale=(x_sample[0] + x_sample[1] + x_sample[2]) / 5,
                )
                for x_sample in X
            ]
        ).astype(dtype)

    return X, Y


""" load parquet dataset """


def load_parquet(file_addresses, read_columns=None):

    table = pa.concat_tables(
        pq.read_table(
            file_address,
            columns=read_columns,
        )
        for file_address in file_addresses
    )

    return table.to_pandas()
