from typing import Optional
import __main__ as main
import datetime

import matplotlib

if hasattr(main, '__file__'):
    matplotlib.use('TkAgg')  # outside of jupyter notebook this backend works best for me 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn import metrics
from sklearn.pipeline import Pipeline
from encoders import ColumnSubset


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


def results_table(y_true: np.ndarray, y_pred: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
    """Creates dataframe with dates, targets and predictions"""
    date = data.apply(lambda x: datetime.datetime(x['year'], x['month'], x['day'], x['hour']), axis=1)
    return pd.DataFrame([v for v in zip(date, y_true, y_pred)],
                        columns=["date", "target", "prediction"], dtype=object).sort_values("date", axis=0)


def plot_target_prediction(y_true: np.ndarray, y_pred: np.ndarray, data: pd.DataFrame,
                           ax: Optional[matplotlib.axes._base._AxesBase] = None) -> None:
    """plotting targets"""
    resdf = results_table(y_true, y_pred, data)
    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 4)
    ax.plot(resdf["date"], resdf[["target", "prediction"]])
    ax.legend(['targets', 'predictions'])
    plt.show()


def print_scores(val_true, val_pred, test_true, test_pred, name=None):
    """computing and printing scores"""

    format_str = "{:>70}: {:4.2f}"

    print('\n\n')

    if (val_pred is not None) and (val_true is not None):
        print(format_str.format(f"{name} validation R2 score ", metrics.r2_score(val_true, val_pred)))
        print(format_str.format(f"{name} validation mean poisson deviance ",
                                metrics.mean_poisson_deviance(val_true, val_pred)))
        print('\n')

    print(format_str.format(f"{name} test R2 score ", metrics.r2_score(test_true, test_pred)))
    print(format_str.format(f"{name} test mean poisson deviance ",
                            metrics.mean_poisson_deviance(test_true, test_pred)))

lookup_estimator = Pipeline([("preprocessing", ColumnSubset(["month", "day", "hour"])),
                             ("model", LookupMeanEstimator())])