import unittest
import os
import pandas as pd
from data_pipeline import get_data, split_data, INTERVAL_DICT, DATA_PATH


def unique_dataframe_rows(df):
    """helper functions returning set of (unique) dataframe rows as tuples"""
    return set([tuple(x) for x in df.values.tolist()])


class DataLoadingTestCase(unittest.TestCase):

    def test_default_path_exists(self):
        """tests if getting data does not fail"""
        self.assertTrue(os.path.exists(DATA_PATH))

    def test_getting_the_data(self):
        """tests if getting data does not fail"""
        try:
            get_data()
        except Exception as e:
            self.fail("Raised {}".format(e.__class__.__name__))


class DataStructureTestCase(unittest.TestCase):
    def setUp(self):
        """load data"""
        self.interval_dict = INTERVAL_DICT
        self.data = get_data(path=DATA_PATH)

    def test_data_intervals(self):
        """tests if all values from columns are in respective intervals defined by INTERVAL_DICT"""
        for col, interval in self.interval_dict.items():
            self.assertGreaterEqual(self.data[col].min(), self.interval_dict[col][0])
            self.assertLessEqual(self.data[col].min(), self.interval_dict[col][1])

    def test_date_uniqueness(self):
        """tests if all rows have unique dates"""
        df = self.data[["year", "month", "day", "hour"]]
        num_unique = len(unique_dataframe_rows(df))
        num_all = df.shape[0]
        self.assertEqual(num_unique, num_all)

    def test_interval_dict_columns_in_datset(self):
        self.assertTrue(set(self.interval_dict.keys()).issubset(set(self.data.columns)))


class SplitDataTestCase(unittest.TestCase):
    def setUp(self):
        """load data"""
        self.data = get_data(path=DATA_PATH)
        self.train_years = list(range(2016, 2021))
        self.test_years = [2021, 2022]

    def test_years_split(self):
        """test if year are splitted into train/validation and test set correctly"""
        (x, y), (xv, yv), (xt, yt) = split_data(self.data,
                                                train_years=self.train_years,
                                                test_years=self.test_years,
                                                val_ratio=0.1,
                                                target_column="count",
                                                columns_to_drop=[])
        self.assertTrue( set(x["year"]).issubset(set(self.train_years)))
        self.assertTrue(set(xv["year"]).issubset(set(self.train_years)))
        self.assertTrue(set(xt["year"]).issubset(set(self.test_years)))


    def test_datasets_not_overlapping(self):
        """test that there are no common elements in train/val/test sets"""
        (x, y), (xv, yv), (xt, yt) = split_data(self.data,
                                                train_years=list(range(2016, 2021)),
                                                test_years=[2021, 2022],
                                                val_ratio=0.1,
                                                target_column="count",
                                                columns_to_drop=[])
        for s1, s2 in [(x, xv), (x, xt), (x, xv)]:
            self.assertEqual(unique_dataframe_rows(s1).intersection(unique_dataframe_rows(s2)), set())

    def test_column_dropping(self):
        """test that both target column and columns to dropped are not present in the output"""
        (x, y), (xv, yv), (xt, yt) = split_data(self.data,
                                                train_years=list(range(2016, 2021)),
                                                test_years=[2021, 2022],
                                                val_ratio=0.1,
                                                target_column="count",
                                                columns_to_drop=["year"])
        for s in (x, xv, xt):
            self.assertNotIn("count", s.columns)
            self.assertNotIn("year", s.columns)


if __name__ == '__main__':
    unittest.main()
