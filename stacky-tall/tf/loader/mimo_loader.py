import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class RegressionMIMOLoader:
    " Class for preparing Multiple Inputs Multiple Output (MIMO) data for training deep learning regression models. "

    def __init__(self, data, split_size, batch_size, window_size, target_names, cache_on = 'RAM'):
        " Initialize the class with the data, split size, batch size, window size, and target name. "
        self.data = data
        self.batch_size = batch_size
        self.split_size = split_size
        self.window_size = window_size
        self.target_names = target_names
        self.cache_on = cache_on  # cache on RAM (small datasets) or not cache

    def _split_data(self):
        " Split the data into training and testing. "
        train, test = train_test_split(self.data, test_size=self.split_size)
        return train, test
        
    def _prepare_data(self, data):
        " Prepare the data for training and validation. "
        # Create a tf.data.Dataset from the pandas dataframe
        features = tf.data.Dataset.from_tensor_slices(data.drop(columns=self.target_names))
        target = tf.data.Dataset.from_tensor_slices(data[self.target_names])
        # Window the features and target
        windowed_features = features.window(self.window_size,  drop_remainder=True)
        windowed_target = target.window(self.window_size,  drop_remainder=True)
        # Flatten the windows
        feat = windowed_features.flat_map(lambda window: window.batch(self.window_size))
        label = windowed_target.flat_map(lambda window: window.batch(self.window_size))
        # Combine the features and target
        dataset = tf.data.Dataset.zip((feat, label))
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
        " Prepare dat for testing"
        features = tf.data.Dataset.from_tensor_slices(test_data)  # features only
        windowed_features = features.window(self.window_size,  drop_remainder=True)  # windowing features
        feat = windowed_features.flat_map(lambda window: window.batch(self.window_size)) # apply flat map
        dataset = tf.data.Dataset.zip(feat)  # zip data
        # Batch the dataset
        if self.cache_on == 'RAM':
            return dataset.batch(self.batch_size).cache().prefetch(tf.data.experimental.AUTOTUNE)
        elif self.cache_on == 'NO_CACHE':
            return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)


if __name__=="__main__":

    # create dataset with features and label
    train_data = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'label1': np.random.rand(100), 'label2': np.random.rand(100)})
    # initialize instance object
    miso = RegressionMIMOLoader(data=train_data, split_size=0.2, batch_size=32, window_size=5, target_names=['label1', 'label2'])

    # process data for training and validation
    train_data, val_data = miso.prepare_training_validation()

    # create test data
    test_data = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100)})

    # process data for testing
    test_data = miso.prepare_test(test_data)
    
    # print shape data
    for x, y in train_data:
        print('training', x.shape, y.shape)
        break

    for x, y in val_data:
        print('validation', x.shape, y.shape)
        break

    for x in test_data:
        print('testing', x.shape)
        break