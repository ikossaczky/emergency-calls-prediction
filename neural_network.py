import tensorflow as tf

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
    from sklearn import metrics
    from data_pipeline import get_data, split_data, OneHotEncoder, PeriodicEncoder, INTERVAL_DICT

    # get the data
    data = get_data()

    # split the data
    (x, y), (xv, yv), (xt, yt) = split_data(data,
                                            train_years=list(range(2016, 2021)),
                                            test_years=[2021, 2022],
                                            val_ratio=0.1,
                                            target_column="count",
                                            columns_to_drop=["year"])

    # we cannot use sklearn pipeline in this case: it does not support e.g. epoch as argument
    encoder = PeriodicEncoder(interval_dict=INTERVAL_DICT)
    x, xv, xt = [encoder.fit_transform(data) for data in [x, xv, xt]]

    # build the model
    net = build_model()

    # fit the network
    net.fit(x, y, epochs=100, validation_data=(xv, yv), callbacks=stopping_callback)


    # print scores of lookup estimator for comparison
    format_str = "{:>40}: {:4.2f}"
    print(format_str.format("validation score lookup estimator", metrics.r2_score(yv, net.predict(xv))))
    print(format_str.format("test score lookup estimator", metrics.r2_score(yt, net.predict(xt))))
