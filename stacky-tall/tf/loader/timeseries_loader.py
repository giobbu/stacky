import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class AutoregressiveMIMOLoader:
    " Class for preparing Multiple Inputs Multiple Output (MIMO) data for training deep learning regression models. "

    def __init__(self, data, split_size, batch_size, window_size_past, window_size_future, cache_on = 'RAM'):
        " Initialize the class with the data, split size, batch size, window size, and target name. "
        self.data = data
        self.batch_size = batch_size
        self.split_size = split_size
        self.window_size_past = window_size_past
        self.window_size_future = window_size_future
        self.cache_on = cache_on  # cache on RAM (small datasets) or not cache

    def _split_data(self):
        " Split the data into training and testing. "
        train, test = train_test_split(self.data, test_size=self.split_size)
        return train, test
        
    def _prepare_data(self, data):
        " Prepare the data for training and validation. "
        # Create a tf.data.Dataset from the pandas dataframe
        dataset = tf.data.Dataset.from_tensor_slices(data)
        # lagged features
        past = dataset.window(self.window_size_past,  shift=1,  stride=1,  drop_remainder=True)
        lagged_feature = past.flat_map(lambda window: window.batch(self.window_size_past))
        # multistep target
        future = dataset.window(self.window_size_future, shift=1,  stride=1,  drop_remainder=True).skip(self.window_size_past)
        multistep_target = future.flat_map(lambda window: window.batch(self.window_size_future))
        # Combine the features and target
        dataset = tf.data.Dataset.zip((lagged_feature, multistep_target))
        if self.cache_on == 'RAM':
            return dataset.batch(self.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        elif self.cache_on == 'NO_CACHE':
            return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    def prepare_training_validation(self):
        " Prepare the data for training and validation. "
        train, test = self._split_data()
        train_data = self._prepare_data(train)
        val_data = self._prepare_data(test)
        return train_data, val_data

    def prepare_test(self, test_data):
        " Prepare data for testing"
        dataset = tf.data.Dataset.from_tensor_slices(test_data)
        # lagged features
        past = dataset.window(self.window_size_past,  shift=1,  stride=1,  drop_remainder=True)
        lagged_feature = past.flat_map(lambda window: window.batch(self.window_size_past))
        # Combine the features and target
        dataset = tf.data.Dataset.zip(lagged_feature)
        if self.cache_on == 'RAM':
            return dataset.batch(self.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        elif self.cache_on == 'NO_CACHE':
            return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)


if __name__=="__main__":

        # UNIVARIATE CASE

    # create dataset with features and label
    train_data = pd.DataFrame({'timeseries': np.random.rand(100)})

    # create an instance of the class
    miso = AutoregressiveMIMOLoader(data=train_data, split_size=0.2, batch_size=32, window_size_past=5, window_size_future=3)

    # prepare training and validation data
    train_data, val_data = miso.prepare_training_validation()

    # prepare test data
    test_data = np.random.rand(100)
    test_data = miso.prepare_test(test_data)

    # plot two consecutive windows
    print('UNIVARIATE CASE')
    for x, y in train_data.take(1):
        for i in range(2):
            print('\n')
            print(f'Window {i+1} features:')
            print(x[i].numpy())
            print('\n')



    # MULTIVARIATE CASE

    # create dataset with features and label
    train_data = pd.DataFrame({'timeseries1': np.random.rand(100), 'timeseries2': np.random.rand(100), 'timeseries3': np.random.rand(100), 'timeseries4': np.random.rand(100)})

    # create an instance of the class
    miso = AutoregressiveMIMOLoader(data=train_data, split_size=0.2, batch_size=32, window_size_past=5, window_size_future=3)

    # prepare training and validation data
    train_data, val_data = miso.prepare_training_validation()

    # prepare test data
    test_data = np.random.rand(100)
    test_data = miso.prepare_test(test_data)

    # plot two consecutive windows
    print('\n')
    print('MULTIVARIATE CASE')
    for x, y in train_data.take(1):
        for i in range(2):
            print('\n')
            print(f'Window {i+1} features:')
            print(x[i].numpy())
            print('\n')
