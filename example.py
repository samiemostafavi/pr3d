
import json
import os
from os.path import abspath, dirname
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from petastorm.spark import SparkDatasetConverter
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

import tensorflow as tf
from petastorm.spark import SparkDatasetConverter
from pr3d.de import ConditionalGaussianMixtureEVM, ConditionalGaussianMM, GaussianMM, GaussianMixtureEVM, GammaMixtureEVM
from pyspark.sql import SparkSession
from pyspark.sql.functions import rand

# requirements: numpy matplotlib loguru seaborn polars SciencePlots pre-commit black flake8 isort codespell scipy petastorm

def main():

    DATA_PATH = './results/example/data'
    CONF_PATH = './example_training_conf.json'
    ORDER_SEED = 12345
    SAMPLE_SEED = 54321
    RES_PRED_PATH = './results/example/model/'

    logger.info("Load dataset and sample")

    # init Spark
    spark = (
        SparkSession.builder.appName("Training")
        .config("spark.driver.memory", "5g")
        .config("spark.driver.maxResultSize", 0)
        .getOrCreate()
    )
    # sc = spark.sparkContext

    # Set a cache directory on DBFS FUSE for intermediate data.
    file_path = dirname(abspath(__file__))
    spark_cash_addr = "file://" + file_path + "/__sparkcache__/__main__"
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, spark_cash_addr)
    logger.info(
        f"load_dataset_and_sample: Spark cache folder is set up at: {spark_cash_addr}"
    )


    # read all the files from the project
    files = []
    logger.info(f"Opening the path {DATA_PATH}")
    all_files = os.listdir(DATA_PATH)
    for f in all_files:
        if f.endswith(".parquet"):
            files.append(DATA_PATH + "/" + f)

    # read all files into one Spark df
    main_df = spark.read.parquet(*files)

    # Absolutely necessary for randomizing the rows (bug fix)
    # first shuffle, then sample!
    main_df = main_df.orderBy(rand(seed=ORDER_SEED))

    # load training params
    with open(CONF_PATH) as json_file:
        model_conf = json.load(json_file)

    training_params = model_conf["training_params"]
    # take the desired number of records for learning
    df_train = main_df.sample(
        withReplacement=False,
        fraction=training_params["dataset_size"] / main_df.count(),
        seed=SAMPLE_SEED,
    )

    logger.info(
        f"Sample {training_params['dataset_size']} rows, result {df_train.count()} samples"
    )

    df_train = df_train.toPandas()

    logger.info(
        f"Training starts with params {model_conf}"
    )

    # set data types
    # npdtype = np.float64
    # tfdtype = tf.float64
    strdtype = "float64"

    logger.info(f"Opening results directory '{RES_PRED_PATH}'")
    os.makedirs(RES_PRED_PATH, exist_ok=True)


    logger.info(
        f"Dataset loaded, train sampels: {len(df_train)}"
    )

    # get parameters
    y_label = model_conf["y_label"]
    model_type = model_conf["type"]
    training_rounds = training_params["rounds"]
    batch_size = training_params["batch_size"]

    if "condition_labels" not in model_conf:

        # dataset pre process
        df_train = df_train[[y_label]]
        df_train["y_input"] = df_train[y_label]
        df_train = df_train.drop(columns=[y_label])

        # initiate the non conditional predictor
        if model_type == "gmm":
            model = GaussianMM(
                centers=model_conf["centers"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"]
            )
        elif model_type == "gmevm":
            model = GaussianMixtureEVM(
                centers=model_conf["centers"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"]
            )

        X = None
        Y = df_train.y_input

    else:
    
        condition_labels = model_conf["condition_labels"]
        # dataset pre process
        df_train = df_train[[y_label, *condition_labels]]
        df_train["y_input"] = df_train[y_label]
        df_train = df_train.drop(columns=[y_label])

        # initiate the non conditional predictor
        if model_type == "gmm":
            model = ConditionalGaussianMM(
                x_dim=condition_labels,
                centers=model_conf["centers"],
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )
        elif model_type == "gmevm":
            model = ConditionalGaussianMixtureEVM(
                x_dim=condition_labels,
                centers=model_conf["centers"],
                hidden_sizes=model_conf["hidden_sizes"],
                dtype=strdtype,
                bayesian=model_conf["bayesian"],
                # batch_size = 1024,
            )

        X = df_train[condition_labels]
        Y = df_train.y_input


    steps_per_epoch = len(df_train) // batch_size

    for idx, round_params in enumerate(training_rounds):

        logger.info(
            "Training session "
            + f"{idx+1}/{len(training_rounds)} with {round_params}, "
            + f"steps_per_epoch: {steps_per_epoch}, batch size: {batch_size}"
        )

        model.training_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=round_params["learning_rate"],
            ),
            loss=model.loss,
        )

        if X is None:
            Xnp = np.zeros(len(Y))
            Ynp = np.array(Y)
            model.training_model.fit(
                x=[Xnp, Ynp],
                y=Ynp,
                steps_per_epoch=steps_per_epoch,
                epochs=round_params["epochs"],
                verbose=1,
            )
        else:
            Xnp = np.array(X)
            Ynp = np.array(Y)
            training_data = tuple([Xnp[:, i] for i in range(len(condition_labels))]) + (
                Ynp,
            )
            model.training_model.fit(
                x=training_data,
                y=Ynp,
                steps_per_epoch=steps_per_epoch,
                epochs=round_params["epochs"],
                verbose=1,
            )

    # training done, save the model
    model.save(RES_PRED_PATH + "model.h5")
    with open(
        RES_PRED_PATH + f"model.json", "w"
    ) as write_file:
        json.dump(model_conf, write_file, indent=4)

    logger.info(
        f"{model_type} {'bayesian' if model.bayesian else 'non-bayesian'} "
        + "model got trained and saved."
    )



if __name__ == "__main__":
    main()
