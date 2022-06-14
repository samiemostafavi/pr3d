# https://keras.io/examples/keras_recipes/bayesian_neural_networks/

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import tensorflow_io as tfio
import matplotlib.pyplot as plt

from bayesian import create_probablistic_bnn_model, negative_loglikelihood, load_bnn_model

tf.keras.backend.set_floatx('float32')

def get_train_and_test_splits(
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
    dataset = (
        tfio.IODataset.from_parquet(
            filename = file_addr,
        )
        .prefetch(buffer_size=dataset_size)
    )

    def read_parquet(features):
        # features is an OrderedDict

        # prepare empty tensors
        keys = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
        values_dict = {}
        
        for a in features.items():
            # look for the features
            for idx,feature_name in enumerate(feature_names):
                # a is a tuple, first item is the key, second is the tensor
                if a[0].decode("utf-8")==feature_name:
                    values = tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                    values = values.write(values.size(), tf.cast(a[1],dtype=dtype))
                    # important to have the squeeze to get (None,) tensor shape
                    values_dict[feature_name] = tf.squeeze(values.stack(),axis=0)

            # look for the keys
            if a[0].decode("utf-8")==label_name:
                keys = keys.write(keys.size(), tf.cast(a[1],dtype=dtype))

        # important to have the squeeze to get (None,) tensor shape
        return (values_dict, tf.squeeze(keys.stack(),axis=0))

    dataset = dataset.map(read_parquet)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).cache().shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).take(dataset_size-train_size).cache().shuffle(buffer_size=train_size).batch(batch_size)

    # to check what is being read:
    #for ds in train_dataset:
    #    print(tfds.as_numpy(ds))
    #for ds in test_dataset:
    #    print(tfds.as_numpy(ds))

    return train_dataset, test_dataset




dataset_size = 8995 #4898
train_size = int(dataset_size * 0.85)
batch_size = 512 #256

feature_names = [
    "queue_length1",
    "queue_length2",
    "queue_length3",
]

label_name = "end2end_delay"

train_dataset, test_dataset = get_train_and_test_splits(
    file_addr = './dataset.parquet',
    feature_names = feature_names,
    label_name = label_name,
    dataset_size = dataset_size,
    train_size = train_size,
    batch_size = batch_size,
)

#print(train_dataset)
#for ds in train_dataset:
#    print(ds)

#print(test_dataset)
#for ds in test_dataset:
#    print(ds)

hidden_units = [8, 8]
prob_bnn_model = create_probablistic_bnn_model(
    train_size = train_size,
    feature_names = feature_names, 
    hidden_units = hidden_units,
    dtype = tf.float32,
)

prob_bnn_model.summary()

learning_rate = 0.005
num_epochs = 5000 #1000

prob_bnn_model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss=negative_loglikelihood,
        metrics=[keras.metrics.KLDivergence()],
    )

print("Start training the model...")
hist = prob_bnn_model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
plt.plot(hist.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Total loss')

prob_bnn_model.save("bnn_threehop_model.h5")
reconstructed_model = load_bnn_model("bnn_threehop_model.h5")

exit(0)

# Extract 100 samples
sample = 100
examples, targets = list(
    test_dataset
    .unbatch()
    .shuffle(buffer_size=batch_size*10)
    .batch(sample)
)[0]

# X:
print(examples)
# Y:
print(targets)

#compute_predictions(examples=examples,model=prob_bnn_model,iterations=100)

#exit(0)
iterations = 2

predicted_distros = []
for _ in range(iterations):
    prediction_distribution = prob_bnn_model(examples)
    predicted_distros.append(prediction_distribution)

    prediction_mean = prediction_distribution.mean().numpy().tolist()
    prediction_stdv = prediction_distribution.stddev().numpy()

    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()

    for idx in range(sample):
        print(
            f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
            f"stddev: {round(prediction_stdv[idx][0], 2)}, "
            f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
            f" - Actual: {targets[idx]}"
        )
