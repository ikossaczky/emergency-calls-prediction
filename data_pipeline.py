import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

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


class BaseEncoder(BaseEstimator, TransformerMixin):
    """
    Base class for data Processing. Also serves as dummy postprocessor returning the dataframe itself.
    Follows the informal sklearn transformer protokol.
    Also, any sklearn transformer can be used as postprocessor.
    """

    def fit(self, df: pd.DataFrame, y=None):
        """no fitting needed for the BaseProcessor"""
        return self

    def transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        return df.values

    def __call__(self, df: pd.DataFrame):
        self.transform(df)


# TODO: test min, max , row_sum, shape
class OneHotEncoder(BaseEncoder):
    """
    Encode each column having n possible values into n columns with one at the position corresponding to the value
    and zero elsewhere
    """

    def __init__(self, interval_dict: dict[str, tuple[int, int]]):
        """
        Initialize OneHotEncoder
        :param interval_dict: dictionary holding for each column name its minimal and maximal values
        """
        self.columns = list(interval_dict.keys())

        # compute interval sizes
        self.interval_lens = [interval[1] - interval[0] + 1 for interval in interval_dict.values()]

        # get intervals minimal values
        self.interval_mins = [interval[0] for interval in interval_dict.values()]

        self.column_onehots_start = np.concatenate([[0], np.cumsum(self.interval_lens[:-1])])

        # construct array of onehot vectors that will be picked from
        self.onehot_vectors = np.eye(np.sum(self.interval_lens))

    def transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        # check if columns from interval_dict can be found in the dataframe
        assert set(self.columns).issubset(set(df.columns)), \
            "columns missing in dataframe: {}".format(list(set(self.columns) - set(df.columns)))

        # convert each column into larger onehot representation and merge
        one_hots = 0
        for col, interval_min, onehot_start, in zip(self.columns, self.interval_mins, self.column_onehots_start):
            one_hots += (self.onehot_vectors[df[col] - interval_min + onehot_start])

        return one_hots


class PeriodicEncoder(BaseEncoder):
    """
    Process features wth sine and cosine. These output feature respect periodical nature of the data.
    E.g. day 365 (31.12.) is very near to day 1 (1.1.), and should therefore have similar feature value
    """

    def __init__(self, interval_dict: dict[str, tuple[int, int]]):
        """
        Initialize PeriodicEncoder
        :param interval_dict: dictionary holding for each column name its minimal and maximal values
        """
        self.columns = list(interval_dict.keys())

        # compute interval sizes
        self.interval_lens = np.array([interval[1] - interval[0] + 1 for interval in interval_dict.values()])[None, :]

        # get intervals minimal values
        self.interval_mins = np.array([interval[0] for interval in interval_dict.values()])[None, :]

    def transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        # check if columns from interval_dict can be found in the dataframe
        assert set(self.columns).issuperset(set(df.columns)), \
            "columns missing in dataframe: {}".format(list(set(self.columns) - set(df.columns)))

        # normalize features so to be from [0,1]
        x = (df[self.columns].values - self.interval_mins) / self.interval_lens

        # create periodic features
        x = np.concatenate([np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)], axis=1)

        return x


class LookupMeanEstimator(BaseEstimator, RegressorMixin):
    """very simple estimator returning mean value of entries having the same features"""

    def __init__(self):
        self.lookup = None

    def fit(self, x, y):
        # create lookup dataframe
        xy = np.concatenate([x, y.reshape(-1, 1)], axis=1)
        self.lookup = pd.DataFrame(xy).groupby(by=list(range(x.shape[1]))).mean()
        return self

    def predict(self, x):
        assert self.lookup is not None, "please fit the LookupMeanEstimator"
        # pick value from lookup dataframe
        return self.lookup.loc[np.array(x).tolist()].iloc[:, 0].values


class ColumnSubset(BaseEncoder):
    """Keeps or drops columns by name"""

    def __init__(self, columns: list[str], drop=False):
        self.columns = columns
        self.drop = drop

    def transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        if self.drop:  # drop columns
            return df.drop(self.columns, axis=1)
        else:  # or pick columns
            return df[self.columns].copy()


# lookup estimator
lookup_estimator = Pipeline([("preprocessing", ColumnSubset(["month", "day", "hour"])),
                             ("model", LookupMeanEstimator())])

# dummy decision tree estimator pipeline = should be +- identical with lookup estimater (due to unconstrained depth)
dummy_estimator = Pipeline([("preprocessing", ColumnSubset(["month", "day", "hour"])),
                            ("model", DecisionTreeRegressor())])

if __name__ == "__main__":
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.pipeline import Pipeline
    from sklearn import metrics

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
    format_str = "{:>40}: {:4.2f}"
    print(format_str.format("model validation score", estimator.score(xv, yv)))
    print(format_str.format("model test score", estimator.score(xt, yt)))

    # fit lookup estimator
    lookup_estimator.fit(x, y)

    # print scores of lookup estimator for comparison
    print(format_str.format("mean-lookup validation score ", lookup_estimator.score(xv, yv)))
    print(format_str.format("mean-lookup test score ", lookup_estimator.score(xt, yt)))

    metrics.mean_poisson_deviance(yv, lookup_estimator.predict(xv))

    def results_table(y_true, y_pred, data):
        date = data.apply(lambda x: datetime.datetime(x['year'], x['month'], x['day'], x['hour']), axis=1)
        return pd.DataFrame([v for v in zip(date, y_true, y_pred)],
                            columns=["date", "target", "prediction"], dtype=object).sort_values("date", axis=0)

    resdf = results_table(yt, estimator.predict(xt), xt)

    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    fig.set_size_inches(15,4)
    ax.plot(resdf["date"], resdf[["target", "prediction"]])
