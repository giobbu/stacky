import pandas as pd

class UnivariateMissingImputer:
    """ Univariate missing value imputation methods """

    def __init__(self, df):
        assert isinstance(df, pd.DataFrame), 'Input must be a pandas DataFrame'
        self.df = df

    def fill_summary_statistics(self, column, method='mean'):
        " Impute missing values with summary statistics"
        if method == 'mean':
            return self.df[column].fillna(self.df[column].mean())
        elif method == 'median':
            return self.df[column].fillna(self.df[column].median())
        elif method == 'mode':
            return self.df[column].fillna(self.df[column].mode()[0])
        else:
            raise ValueError('Method not supported')
        
    def fill_constant(self, column, value):
        " Impute missing values with a constant"
        return self.df[column].fillna(value)

    def fill_backfill(self, column):
        " Impute missing values with the next valid observation"
        return self.df[column].fillna(method='backfill')
    
    def fill_forwardfill(self, column):
        " Impute missing values with the last valid observation"
        return self.df[column].fillna(method='ffill')

    def fill_most_frequent(self, column):
        " Impute missing values with the most frequent value"
        return self.df[column].fillna(self.df[column].value_counts().index[0])

    def fill_rolling_statistics(self, column, window=3, method='mean'):
        " Impute missing values with rolling summary statistics"
        if method == 'mean':
            return self.df[column].fillna(self.df[column].rolling(window=window, min_periods=1).mean())
        elif method == 'median':
            return self.df[column].fillna(self.df[column].rolling(window=window, min_periods=1).median())
        else:
            raise ValueError('Method not supported')
        
    def fill_interpolate(self, column, method='linear', order=0):
        """Impute missing values with interpolation."""
        supported_methods = ['linear', 'quadratic', 'cubic', 'polynomial', 'spline']
        assert method in supported_methods, f'Method {method} not supported. Supported methods: {supported_methods}'
        if method in ['polynomial', 'spline']:
            assert order > 0, 'Order must be greater than 0 for polynomial or spline interpolation'
        if method == 'polynomial' or method == 'spline':
            return self.df[column].interpolate(method=method, order=order)
        else:
            return self.df[column].interpolate(method=method)

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