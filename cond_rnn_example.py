import random
import tensorflow as tf
import numpy as np
from loguru import logger
from pyspark.sql import SparkSession
from pr3d.de import ConditionalRecurrentGaussianMM


enable_training = False
parquet_file = "10-42-3-2_55500_20230726_171830.parquet"
target_var = "delay.send"
covariate = "netinfodata.CSQ"

spark = (
    SparkSession.builder.master("local")
    .appName("LoadParquets")
    .config("spark.executor.memory", "6g")
    .config("spark.driver.memory", "70g")
    .config("spark.driver.maxResultSize", 0)
    .getOrCreate()
)

# find dataframe with the desired condition
# inputs: exp_args["condition_nums"]
df = spark.read.parquet(parquet_file)
total_count = df.count()
logger.info(f"Parquet file {parquet_file} is loaded.")
logger.info(f"Total number of samples in this empirical dataset: {total_count}")

# preprocess latency
measurements = df.rdd.map(lambda x: x[target_var]).collect()
time_series_data = np.array(measurements)/1e6

# preprocess CSQ covariate
covariate_csq = df.rdd.map(lambda x: x[covariate]).collect()
covariate_csq = np.array(covariate_csq)
covariate_csq = covariate_csq.astype(np.float64)


if enable_training:

    recurrent_taps = 64
    epochs = 20
    batch_size = 128
    num_training_samples = 10000

    model = ConditionalRecurrentGaussianMM(
        centers=8,
        x_dim=["CSQ"],
        recurrent_taps=recurrent_taps,
        hidden_layers_config={ 
            "hidden_lstm_1": { "type": "lstm","size": 64, "return_sequences": True },
            "hidden_lstm_2": { "type": "lstm","size": 64, "return_sequences": False },
            "hidden_dense_1": { "type": "dense","size": 16,"activation": "tanh" },
            "hidden_dense_2": { "type": "dense","size": 16,"activation": "tanh" }, 
        }
    )

    # limit the number of samples
    time_series_data = time_series_data[:num_training_samples]
    logger.info(f"Limited the number of samples for training: {len(time_series_data)}")

    # number of taps
    num_taps = recurrent_taps

    # Create input (Y) and target (y) data
    Y, y, X = [], [], []
    for i in range(len(time_series_data) - num_taps):
        # target sequence
        Y.append(time_series_data[i:i+num_taps])
        # covariate sequence
        X.append(covariate_csq[i:i+num_taps])
        # target value
        y.append(time_series_data[i+num_taps])
    Y = np.array(Y)
    X = np.array(X)
    y = np.array(y)

    # Reshape the input data for LSTM (samples, time steps, features)
    Y = Y.reshape(Y.shape[0], num_taps, 1)
    X = X.reshape(X.shape[0], num_taps, 1)

    # Split the data into training and testing sets (adjust the split ratio as needed)
    split_ratio = 0.8
    split_index = int(len(Y) * split_ratio)
    Y_train, Y_test = Y[:split_index], Y[split_index:]
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    logger.info(f"Number of training sequences: {len(y_train)}")
    logger.info(f"Number of test sequences: {len(y_test)}")

    model.training_model.compile(
        optimizer=tf.keras.optimizers.Adam(
            #learning_rate=learning_rate,
        ),
        loss=model.loss,
    )

    # find steps_per_epoch
    steps_per_epoch = len(y_train) // batch_size
    # train!
    model.training_model.fit(
        x=[Y_train, X_train, y_train],
        y=y_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1,
    )

    # Evaluate the model
    loss = model.training_model.evaluate(
        x=[Y_test, X_test, y_test], 
        y=y_test
    )
    print(f"Test Loss: {loss}")

    # Save the model to a file
    # training done, save the model
    model.save("model_cond_rnn_gmm.h5")
    logger.success("Model saved successfully.")


model = ConditionalRecurrentGaussianMM(h5_addr="model_cond_rnn_gmm.h5")
model.core_model._model.summary()
logger.success("Model loaded successfully.")

# number of taps
num_taps = model.recurrent_taps
print(num_taps)

# Create input (Y) and target (y) data
Y, y, X = [], [], []
for i in range(len(time_series_data) - num_taps):
    # target sequence
    Y.append(time_series_data[i:i+num_taps])
    # covariate sequence
    X.append(covariate_csq[i:i+num_taps])
    # target value
    y.append(time_series_data[i+num_taps])
Y = np.array(Y)
X = np.array(X)
y = np.array(y)


# select a single sequence and check probability
indx = random.choice(range(Y.shape[0]))
singleY = Y[indx,:]
singleX = X[indx,:]
singley = 10 #ms
logger.info(f"check the probability of X:{singleX}, Y:{singleY} at {singley} ms")
result = model.prob_single(singleY,singleX,singley)
logger.success(f"pdf:{result[0]}, log_pdf:{result[1]}, ecdf:{result[2]}")


# use the previous sequences and sample the resulting distribution 20 times in parallel
logger.info(f"produce 20 parallel samples from X:{singleX}, Y:{singleY}")
result = model.sample_n_parallel(singleY,singleX,20)
logger.success(f"parallel samples: {result}")

# select a batch of sequences and a batch of targets, print the result
batch_size = 8
indxes = random.choices(range(Y.shape[0]),k=batch_size)
batchY = Y[indxes,:]
batchX = X[indxes,:]
batchy = np.array([10,12,14,16,18,20,22,24])
logger.info(f"check the probabilities of a batch of size {batch_size}, X:{batchX}, Y:{batchY} at {batchy} ms")
result = model.prob_batch(batchY,batchX,batchy,batch_size=batch_size)
logger.success(f"pdf:{result[0]}, log_pdf:{result[1]}, ecdf:{result[2]}")
