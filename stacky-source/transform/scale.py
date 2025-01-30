from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from loguru import logger

class SklearnScaler:
    " Scale the data using the sklearn StandardScaler "

    def __init__(self, data):
        self.data = data

    def fit_standard_scaler(self):
        self.scaler = StandardScaler().fit(self.data)
        logger.info(f"Fitting {self.scaler.__class__.__name__}")
        logger.info(f"Mean: {self.scaler.mean_}")
        logger.info(f"Scale: {self.scaler.scale_}")
        return self
    
    def fit_minmax_scaler(self):
        self.scaler = MinMaxScaler().fit(self.data)
        logger.info(f"Fitting {self.scaler.__class__.__name__}")
        logger.info(f"Data Min: {self.scaler.data_min_}")
        logger.info(f"Data Max: {self.scaler.data_max_}")
        return self
    
    def fit_box_cox(self):
        assert (self.data >= 0).all(), "Data must be positive for Box-Cox transformation"
        self.scaler = PowerTransformer(method='box-cox').fit(self.data)
        logger.info(f"Fitting {self.scaler.__class__.__name__}")
        return self
    
    def fit_yeo_johnson(self):
        self.scaler = PowerTransformer(method='yeo-johnson').fit(self.data)
        logger.info(f"Fitting {self.scaler.__class__.__name__}")
        return self
    
    def forward(self, type='standard'):
        if type == 'standard':
            self.fit_standard_scaler()
            scaled_data = self.scaler.transform(self.data)
        elif type == 'minmax':
            self.fit_minmax_scaler()
            scaled_data = self.scaler.transform(self.data)
        elif type == 'box-cox':
            self.fit_box_cox()
            scaled_data = self.scaler.transform(self.data)
        elif type == 'yeo-johnson':
            self.fit_yeo_johnson()
            scaled_data = self.scaler.transform(self.data)

        scaled_data = self.scaler.transform(self.data)
        logger.info(f"Transforming ({self.scaler.__class__.__name__})")
        return scaled_data
    
    def backward(self, scaled_data):
        unscaled_data = self.scaler.inverse_transform(self.data)
        logger.info(f"Inverting transformation ({self.scaler.__class__.__name__})")
        return unscaled_data
    

if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)
    data = np.random.rand(10, 2)
    print(data)
    scaler = SklearnScaler(data)
    scaled_data = scaler.forward(type='minmax')
    print(scaled_data)
    unscaled_data = scaler.backward(scaled_data)
    print(unscaled_data)