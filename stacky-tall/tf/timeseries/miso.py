import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class MISOLoader:
    " Class for preparing Multiple Inputs Single Output (MISO) data for training deep learning models. "

    def __init__(self, data, batch_size, window_size, target_name):
        " Initialize the class with the data, batch size, window size, and target name. "
        self.data = data
        self.batch_size = batch_size
        self.window_size = window_size
        self.target_name = target_name
        
    def prepare_training(self):
        " Prepare the data for training. "
        # Create a tf.data.Dataset from the pandas dataframe
        features = tf.data.Dataset.from_tensor_slices(self.data.drop(columns=[self.target_name]))
        target = tf.data.Dataset.from_tensor_slices(self.data[self.target_name])
        # Window the features and target
        windowed_features = features.window(self.window_size,  drop_remainder=True)
        windowed_target = target.window(self.window_size,  drop_remainder=True)
        # Flatten the windows
        feat = windowed_features.flat_map(lambda window: window.batch(self.window_size))
        label = windowed_target.flat_map(lambda window: window.batch(self.window_size))
        # Combine the features and target
        dataset = tf.data.Dataset.zip((feat, label)) 
        return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    def prepare_test(self):
        " Prepare dat for testing"
        features = tf.data.Dataset.from_tensor_slices(self.data)  # features only
        windowed_features = features.window(self.window_size,  drop_remainder=True)  # windowing features
        feat = windowed_features.flat_map(lambda window: window.batch(self.window_size)) # apply flat map
        dataset = tf.data.Dataset.zip(feat)  # zip data 
        return dataset.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)


if __name__=="__main__":

    # create dataset with features and label
    data = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'label': np.random.rand(100)})

    # split data in traininig and testing
    train, test = train_test_split(data, test_size=0.2)

    # get x_train, y_train, x_test, y_test
    x_train, y_train = train.drop(columns=['label']), train['label']
    x_test, y_test = test.drop(columns=['label']), test['label']

    # initialize instance object
    miso = MISOLoader(data=train, batch_size=32, window_size=5, target_name='label')

    # process data for training
    train_data = miso.prepare_training()

    # process data for testing
    test_data = miso.prepare_test()
    
    # print shape data
    for x, y in train_data:
        print(x.shape, y.shape)
        break

    for x in test_data:
        print(x.shape)
        break