import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin


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
