import numpy as np
import pandas as pd
from collections.abc import Iterable
from typing import Union

DATA_PATH = "./data/dataset.csv"
# TODO: test if data is really from the intervals
INTERVAL_DICT = {"month": (1, 12),
                 "day": (1, 31),
                 "hour": (0, 23),
                 "weekday": (1, 7),
                 "week": (1, 53),
                 "quarter": (1, 4)}


def get_data(path: str = DATA_PATH, dtype: type = np.int16):
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


# TODO: test if data is splited correctly (corect years in trian and val and test set), matching of shapes
def split_data(data: pd.DataFrame,
               test_years: list[int],
               train_years: list[int],
               val_ratio: float):
    test_set = data[data["year"].isin(test_years)].reset_index(drop=True)
    train_val_set = data[data["year"].isin(train_years)].reset_index(drop=True)

    num_val_samples = round(train_val_set.shape[0] * val_ratio)
    val_samples = np.random.choice(train_val_set.shape[0], size=num_val_samples, replace=False)
    train_samples = list(set(range(train_val_set.shape[0])) - set(val_samples))

    train_set = train_val_set.iloc[train_samples, :].reset_index(drop=True)
    val_set = train_val_set.iloc[val_samples, :].reset_index(drop=True)

    return train_set, val_set, test_set


class BaseProcessor:
    """
    Base class for data Processing. Also serves as dummy postprocessor returning the dataframe itself.
    Follows the informal sklearn transformer protokol.
    Also, any sklearn transformer can be used as postprocessor.
    """

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return df.values

    def __call__(self, df: pd.DataFrame):
        self.transform(df)


# TODO: test min, max , row_sum, shape
class OneHotEncoder(BaseProcessor):
    def __init__(self, interval_dict: dict[str, tuple[int, int]]):
        self.max_dict = interval_dict
        self.columns = list(interval_dict.keys())
        self.interval_lens = [interval[1] - interval[0] + 1 for interval in interval_dict.values()]
        self.interval_mins = [interval[0] for interval in interval_dict.values()]

        self.column_onehots_start = np.concatenate([[0], np.cumsum(self.interval_lens[:-1])])

        self.onehot_vectors = np.eye(np.sum(self.interval_lens))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert set(self.columns).issubset(set(df.columns)), \
            "columns missing in dataframe: {}".format(list(set(self.columns) - set(df.columns)))

        one_hots = 0
        for col, interval_min, onehot_start, in zip(self.columns, self.interval_mins, self.column_onehots_start):
            one_hots += (self.onehot_vectors[df[col] - interval_min + onehot_start])

        return one_hots


class DataPipeline:
    """
    DataPipeline prepares the data for training or prediction:
    """

    def __init__(self,
                 target_column: str,
                 columns_to_drop: list[str],
                 processors: Union[list[BaseProcessor], BaseProcessor, None]):

        """
        :param target_column: which data columns should be used as target
        :param columns_to_drop: which other columns should be dropped from the training data
        :param postprocessor: optional function which can further process the model input
        """
        self.target_column = target_column
        self.columns_to_drop = columns_to_drop
        self.processors = processors

        if self.processors is None:
            self.processors = [BaseProcessor()]

        if not isinstance(self.processors, Iterable):
            self.processors = [processors]

        # target column should be dropped from the input data in any case
        if not self.target_column in self.columns_to_drop:
            self.columns_to_drop.append(self.target_column)

    def _input_target_split(self, data: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        y = data[self.target_column].values
        x = data.drop(self.columns_to_drop, axis=1)
        return x, y

    def fit(self, data: pd.DataFrame):
        x, _ = self._input_target_split(data)
        for processor in self.processors:
            if hasattr(processor, "fit"):
                processor.fit(x)

    def transform(self, data: pd.DataFrame) -> tuple[Union[list[np.ndarray], np.ndarray], np.ndarray]:
        x, y = self._input_target_split(data)
        x_transformed = [processor.transform(x) for processor in self.processors]
        if len(x_transformed) == 1:
            x_transformed = x_transformed[0]
        return x_transformed, y

    def __call__(self, data):
        return self.transform(data)


# y = merged.loc[:, "count"].values
#
# xdf = merged.drop(["year", "count"], axis=1)
#
# encoder = OneHotEncoder(INTERVAL_DICT)
# x = encoder.transform(xdf)

if __name__ == "__main__":

    data = get_data()

    train_df, val_df, test_df = split_data(data, test_years=[2021, 2022],train_years=list(range(2017, 2021)),
                                           val_ratio=0.1)

    # data_pipeline = DataPipeline(target_column="count", processors=OneHotEncoder(INTERVAL_DICT))
    data_pipeline = DataPipeline(target_column="count", columns_to_drop=["year"], processors=BaseProcessor())

    data_pipeline.fit(train_df)

    (x, y), (xv, yv), (xt, yt) = [data_pipeline(df) for df in [train_df, val_df, test_df]]

    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.pipeline import Pipeline
    models = [AdaBoostRegressor(), RandomForestRegressor(max_depth=5), GradientBoostingRegressor(), DecisionTreeRegressor()]
    # for model in models:
    #     print(model.fit(x,y).score(xv, yv))

    df = pd.DataFrame(x)
    df["y"] = y

    model = models[0]
    pip = Pipeline([("data_pipeline", data_pipeline), ("model", model)])
    pip.fit(x,y)

    # model = StackingRegressor(estimators=[(m.__class__.__name__, m) for m in models])
    # print(model.fit(x,y).score(xv,yv))


    # print(merged.max(axis=0))
    # print(merged.min(axis=0))

    # max_dict = {"month"}

    # todo substract min
    # todo add column names (in  bash script)
    # todo TESTS: if data has the right intervals
    # insert hours where nothing happened (how?)
    # date to weekday
    # maybe holidays?
    # weather data?
