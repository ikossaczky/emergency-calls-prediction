import numpy as np
import pandas as pd


from encoders import ColumnSubset, PeriodicEncoder, LookupMeanEstimator

DATA_PATH = "./data/dataset.csv"
# TODO: test if data is really from the intervals
INTERVAL_DICT = {"month": (1, 12),
                 "day": (1, 31),
                 "hour": (0, 23),
                 "weekday": (1, 7),
                 "week": (1, 53),
                 "quarter": (1, 4),
                 "yearday": (1, 366)}


def get_data(path: str = DATA_PATH, dtype: type = np.int16) -> pd.DataFrame:
    """
    Loads the data and adds missing rows corresponding to hours when no emergency call was made.
    These data rows are added only for days present in the dataset: if a whole day is missing we assume it to be a
    logging problem and do not add the da with 0 calls.
    :param path: path to data
    :param dtype: dtype of the data
    :return: DataFrame with the data
    """

    # Load the data. the data is assumed to be sorted by date (process_data.sh guarantees it).
    df = pd.read_csv(path, dtype=dtype, index_col=False, sep=",", encoding="utf8")

    # Data includes only hours were at least one call was made. We will now add also hours when no call was made
    # Get unique days (if some day is not present at all it might also be a logging problem and we will not add it)
    df2 = df.drop(["hour", "count"], axis=1).drop_duplicates()

    # add 24 hours to each unique day
    num_days = df2.shape[0]
    df2 = df2.loc[df2.index.repeat(24)].reset_index(drop=True)
    df2["hour"] = np.array(list(range(24)) * num_days, dtype=dtype)

    # cut out first and last 24 hours as we do not have portions of the data for first and last day
    df2 = df2.iloc[24:-24, :]

    # the original and this new dataframe will be merged based on all time data
    labels_for_merge = list(df.columns)
    labels_for_merge.remove("count")

    # merge frames, replace NaN with zeros: NaNs are in the count column for rows,
    # which were not present in original data datframe, that means when no call was made
    merged = df.merge(df2, how="outer", on=labels_for_merge).fillna(0).astype(dtype)

    return merged


def split_data(data: pd.DataFrame,
               train_years: list[int],
               test_years: list[int],
               val_ratio: float,
               target_column: str,
               columns_to_drop: list[str],
               ) -> tuple[tuple[pd.DataFrame, np.ndarray], ...]:
    """
    Splits the data into (input, target) tuples for training, validation and testing
    :param data: data
    :param train_years: list of years that will be used for training
    :param test_years: list of years that will be used for testing
    :param val_ratio: ratio of the validation set
    :param target_column: name of the column that should be used as target
    :param columns_to_drop: which other columns should be dropped
    :return: tuple of tuples ((x_train, y_train), (x_val, y_val), (x_test, y_test)) where x denotes input
    and y denotes target
    """

    def _input_target_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        # split out target column
        y = data[target_column].values
        # drop defined columns, including target column from the input
        if target_column not in columns_to_drop:
            columns_to_drop.append(target_column)
        x = data.drop(columns_to_drop, axis=1)
        return x, y

    # split data into test set and train/validation set and reset index
    test_set = data[data["year"].isin(test_years)].reset_index(drop=True)
    train_val_set = data[data["year"].isin(train_years)].reset_index(drop=True)

    # randomly pick validation set / remaining samples form train set
    num_val_samples = round(train_val_set.shape[0] * val_ratio)
    val_samples = np.random.choice(train_val_set.shape[0], size=num_val_samples, replace=False)
    train_samples = list(set(range(train_val_set.shape[0])) - set(val_samples))

    # reset index
    train_set = train_val_set.iloc[train_samples, :].reset_index(drop=True)
    val_set = train_val_set.iloc[val_samples, :].reset_index(drop=True)

    return tuple(_input_target_split(df) for df in (train_set, val_set, test_set))



if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.pipeline import Pipeline
    from eval_tools import plot_target_prediction, results_table, print_scores, lookup_estimator

    # get the data
    data = get_data()

    # split the data into input and target for train, validation and test set
    (x, y), (xv, yv), (xt, yt) = split_data(data,
                                            train_years=list(range(2016, 2021)),
                                            test_years=[2021, 2022],
                                            val_ratio=0.1,
                                            target_column="count",
                                            columns_to_drop=[])

    # initialize the model
    model = GradientBoostingRegressor()

    # initialize the data encoder
    columnsubset = ColumnSubset(columns=["year"], drop=True)

    # initialize the data encoder
    encoder = PeriodicEncoder(interval_dict=INTERVAL_DICT)

    # build pipeline encoder + model
    estimator = Pipeline([
        ("select_columns", columnsubset),
        ("encoder", encoder),
        ("model", model)])

    # fit the pipeline
    estimator.fit(x, y)

    # print scores
    yv_pred = estimator.predict(xv)
    yt_pred = estimator.predict(xt)
    print_scores(yv, yv_pred, yt, yt_pred, "GradientBoostingRegressor")

    # fit lookup estimator
    lookup_estimator.fit(pd.concat([x, xv], axis=0), np.concatenate([y, yv], axis=0))

    # print scores of lookup estimator for comparison
    yt_pred = lookup_estimator.predict(xt)
    print_scores(None, None, yt, yt_pred, "lookup_estimator")  # lookup estimator sees validation set

    # save results to csv
    resdf = results_table(yt, estimator.predict(xt), xt)
    resdf.to_csv("./data/results_gradient_boosting.csv")

    # plot predictions and target
    plot_target_prediction(yt, estimator.predict(xt), xt)
