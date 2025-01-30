import missingno as msno
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

class MissingVisualize:
    def __init__(self, df):
        self.df = df

    def matrix(self, sample=100):
        figure = plt.figure(figsize=(10, 5))
        msno.matrix(self.df.sample(sample))
        plt.show()
    
    def bar(self):
        figure = plt.figure(figsize=(10, 5))
        msno.bar(self.df)
        plt.show()

    def heatmap(self):
        "measures nullity correlation between every pair of columns"
        figure = plt.figure(figsize=(10, 5))
        msno.heatmap(self.df)
        plt.show()

if __name__ == '__main__':
    # create 1000x5 dataframe with 10% missing values
    df = pd.DataFrame(np.random.rand(1000, 5))
    df.columns = ['A', 'B', 'C', 'D', 'E']
    df[df < 0.1] = np.nan
    mv = MissingVisualize(df)
    mv.matrix()
    mv.bar()
    mv.heatmap()

    