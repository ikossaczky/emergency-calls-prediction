import numpy as np
import pandas as pd

DATA = "./data/dataset.csv"
dtype = np.int16

# load data
df = pd.read_csv(DATA, dtype=dtype, index_col=False, sep=",", encoding="utf8")

df2 = df.drop(["hour", "count"], axis=1).drop_duplicates()
num_days = df2.shape[0]

df2 = df2.loc[df2.index.repeat(24)].reset_index(drop=True)
df2["hour"] = np.array(list(range(24)) * num_days, dtype=dtype)

# cut out first and last day as these were not fully present in original dataframe
df2 = df2.iloc[24:-24, :]

labels_for_merge = list(df.columns)
labels_for_merge.remove("count")
# pd.concat([df, df2]).groupby(c)

merged = df.merge(df2, how="outer", on=labels_for_merge).fillna(0).astype(dtype)


class DataGenerator:
    def __init__(self, data: pd.DataFrame,
                 batchsize: int,
                 test_years: list[int],
                 train_years: list[int],
                 validation_ratio: float):
        self.data = data
        self.batchsize = batchsize
        self.test_year = test_years
        self.train_years = train_years
        self.validation_ratio = validation_ratio

        self.test_set = self.data[self.data["year"].isin(test_years)].reset_index(drop=True)
        train_val_set = self.data[self.data["year"].isin(train_years)].reset_index(drop=True)

        num_val_samples = train_val_set.shape[0] // self.validation_ratio
        val_samples = np.random.choice(train_val_set.shape[0], size=num_val_samples, replace=False)
        train_samples = list(set(range(train_val_set.shape[0])) - set(val_samples))

        self.val_set = train_val_set.iloc[val_samples, :].reset_index()
        self.train_set = train_val_set.iloc[val_samples, :].reset_index()
        self.sample_order = []


class OneHotPostprocessor():
    def __init__(self, interval_dict: dict[str, tuple[int, int]]):
        self.max_dict = interval_dict
        self.columns, = list(interval_dict.keys())
        self.interval_lens = [interval[1] - interval[0] for interval in interval_dict.values()]
        self.interval_mins = [interval[0] for interval in interval_dict.values()]

        self.column_onehots_start = np.concatenate([0], np.cumsum(self.interval_lens)[1:])

        self.onehot_vectors = np.eye(np.sum(self.interval_lens))

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert set(self.columns) == set(df.columns), "unknown or missing columns in input dataframe"
        one_hots = []
        for col, interval_min, onehot_start, in zip(self.columns, self.interval_mins, self.column_onehots_start):
            one_hots.append(self.onehot_vectors[df["col"] - interval_min + onehot_start])
        return np.concatenate(one_hots, axis=1)

    y = merged.loc[:, "count"].values

    xdf = merged.drop(["year", "count"], axis=1)

    INTERVAL_DICT = {"month": (1, 12),
                     "day": (1, 365),
                     "hour": (0, 23),
                     "week": (1, 53),
                     "weekday": (1, 7)}

    # max_dict = {"month"}

# todo substract min
# todo add column names (in  bash script)
# todo TESTS: if data has the right intervals
# insert hours where nothing happened (how?)
# date to weekday
# maybe holidays?
# weather data?
