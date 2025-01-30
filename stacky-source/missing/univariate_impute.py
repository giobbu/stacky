import pandas as pd
from typing import Union, Literal

class UnivariateMissingImputer:
    """Univariate missing value imputation methods."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the UnivariateMissingImputer.

        :param df: Input pandas DataFrame.
        """
        assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
        self.df = df

    def fill_summary_statistics(
        self, column: str, method: Literal["mean", "median", "mode"] = "mean"
    ) -> pd.Series:
        """
        Impute missing values with summary statistics.

        :param column: Column name to impute.
        :param method: Imputation method ('mean', 'median', or 'mode').
        :return: Series with imputed values.
        """
        if method == "mean":
            return self.df[column].fillna(self.df[column].mean())
        elif method == "median":
            return self.df[column].fillna(self.df[column].median())
        elif method == "mode":
            return self.df[column].fillna(self.df[column].mode()[0])
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'mean', 'median', or 'mode'.")

    def fill_constant(self, column: str, value: Union[int, float, str]) -> pd.Series:
        """
        Impute missing values with a constant.

        :param column: Column name to impute.
        :param value: Constant value to use for imputation.
        :return: Series with imputed values.
        """
        return self.df[column].fillna(value)

    def fill_backfill(self, column: str) -> pd.Series:
        """
        Impute missing values with the next valid observation.

        :param column: Column name to impute.
        :return: Series with imputed values.
        """
        return self.df[column].fillna(method="backfill")

    def fill_forwardfill(self, column: str) -> pd.Series:
        """
        Impute missing values with the last valid observation.

        :param column: Column name to impute.
        :return: Series with imputed values.
        """
        return self.df[column].fillna(method="ffill")

    def fill_most_frequent(self, column: str) -> pd.Series:
        """
        Impute missing values with the most frequent value.

        :param column: Column name to impute.
        :return: Series with imputed values.
        """
        return self.df[column].fillna(self.df[column].value_counts().index[0])

    def fill_rolling_statistics(
        self, column: str, window: int = 3, method: Literal["mean", "median"] = "mean"
    ) -> pd.Series:
        """
        Impute missing values with rolling summary statistics.

        :param column: Column name to impute.
        :param window: Rolling window size.
        :param method: Rolling method ('mean' or 'median').
        :return: Series with imputed values.
        """
        if method == "mean":
            return self.df[column].fillna(self.df[column].rolling(window=window, min_periods=1).mean())
        elif method == "median":
            return self.df[column].fillna(self.df[column].rolling(window=window, min_periods=1).median())
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'mean' or 'median'.")

    def fill_interpolate(
        self, column: str, method: Literal["linear", "quadratic", "cubic", "polynomial", "spline"] = "linear", order: int = 0
    ) -> pd.Series:
        """
        Impute missing values with interpolation.

        :param column: Column name to impute.
        :param method: Interpolation method ('linear', 'quadratic', 'cubic', 'polynomial', or 'spline').
        :param order: Order of interpolation (required for 'polynomial' and 'spline').
        :return: Series with imputed values.
        """
        supported_methods = ["linear", "quadratic", "cubic", "polynomial", "spline"]
        if method not in supported_methods:
            raise ValueError(f"Method '{method}' not supported. Supported methods: {supported_methods}")

        if method in ["polynomial", "spline"] and order <= 0:
            raise ValueError("Order must be greater than 0 for polynomial or spline interpolation.")

        return self.df[column].interpolate(method=method, order=order if method in ["polynomial", "spline"] else None)

if __name__=='__main__':
    # create 1000x5 dataframe with 10% missing values
    import numpy as np
    df = pd.DataFrame(np.random.rand(1000, 6))
    df.columns = ['A', 'B', 'C', 'D', 'E', 'F']
    df[df < 0.1] = np.nan
    imputer = UnivariateMissingImputer(df)
    df['A_fill_mean'] = imputer.fill_summary_statistics('A', method='mean')
    df['A_fill_median'] = imputer.fill_summary_statistics('A', method='median')
    df['A_fill_mode'] = imputer.fill_summary_statistics('A', method='mode')
    df['A_fill_constant'] = imputer.fill_constant('A', value=0)
    df['A_fill_backfill'] = imputer.fill_backfill('A')
    df['A_fill_forwardfill'] = imputer.fill_forwardfill('A')
    df['A_fill_most_frequent'] = imputer.fill_most_frequent('A')
    df['A_fill_rolling_mean'] = imputer.fill_rolling_statistics('A', method='mean')
    df['A_fill_rolling_median'] = imputer.fill_rolling_statistics('A', method='median')
    df['A_fill_linear'] = imputer.fill_interpolate('A', method='linear')
    df['A_fill_quadratic'] = imputer.fill_interpolate('A', method='quadratic')
    df['A_fill_cubic'] = imputer.fill_interpolate('A', method='cubic')
    df['A_fill_polynomial'] = imputer.fill_interpolate('A', method='polynomial', order=2)
    df['A_fill_spline'] = imputer.fill_interpolate('A', method='spline', order=2)
    print(df.head())