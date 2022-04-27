import tensorflow as tf
import numpy as np
import pandas as pd

# Callback for early stopping if validation loss does not improve anymore
stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=3,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)


# Function that builds and returns the neural network
def build_model():
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Dense(8, activation='elu'))
    net.add(tf.keras.layers.Dense(4, activation='elu'))
    net.add(tf.keras.layers.Dense(1, activation='relu'))

    net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    return net


if __name__ == "__main__":
    from data_pipeline import get_data, split_data, INTERVAL_DICT
    from encoders import PeriodicEncoder, OneHotEncoder, ColumnSubset
    from eval_tools import plot_target_prediction, results_table, print_scores, lookup_estimator


    # get the data
    data = get_data()

    # split the data
    (x, y), (xv, yv), (xt, yt) = split_data(data,
                                            train_years=list(range(2016, 2021)),
                                            test_years=[2021, 2022],
                                            val_ratio=0.05,
                                            target_column="count",
                                            columns_to_drop=[])

    # save original data for lookup estimator for comparison
    x_orig, xv_orig, xt_orig = x, xv, xt

    # I don't want to use sklearn pipeline for keras model as it has quite bad support, and a lot of hacks are needed:
    colsubset = ColumnSubset(columns=["year"], drop=True)
    x, xv, xt = [colsubset.fit_transform(data) for data in [x, xv, xt]]

    encoder = PeriodicEncoder(interval_dict=INTERVAL_DICT)
    x, xv, xt = [encoder.fit_transform(data) for data in [x, xv, xt]]

    # build the model
    estimator = build_model()

    # fit the network
    estimator.fit(x, y, epochs=100, validation_data=(xv, yv), callbacks=stopping_callback)

    # print scores
    yv_pred = estimator.predict(xv)
    yt_pred = estimator.predict(xt)
    print_scores(yv, yv_pred, yt, yt_pred, "neural network")

    # fit lookup estimator for comparison
    lookup_estimator.fit(pd.concat([x_orig, xv_orig], axis=0), np.concatenate([y, yv], axis=0))

    # print scores of lookup estimator for comparison
    yt_pred = lookup_estimator.predict(xt_orig)
    print_scores(None, None, yt, yt_pred, "lookup_estimator")  # lookup estimator sees validation set

    # save results to csv
    resdf = results_table(yt, estimator.predict(xt), xt_orig)
    resdf.to_csv("./data/results_neural_network.csv")

    # plot predictions and target
    plot_target_prediction(yt, estimator.predict(xt), xt_orig)


