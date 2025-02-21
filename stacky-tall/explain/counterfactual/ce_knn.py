import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from loguru import logger

class KNNCounterfactualGenerator:
    """Generate counterfactuals for a given instance."""
    
    def __init__(self, instance, data, labels, k=5, metric='euclidean', direction=None, prob_direction=None, classification_type=None):
        assert isinstance(instance, np.ndarray), "Instance should be a numpy array."
        assert instance.ndim == 1, "Instance should be a 1D array."
        assert isinstance(data, np.ndarray), "Data should be a numpy array."
        assert data.ndim == 2, "Data should be a 2D array."
        assert isinstance(labels, np.ndarray), "Labels should be a numpy array."
        assert instance.shape[0] == data.shape[1], "The number of features in the instance should match the data."
        assert isinstance(k, int), "k should be an integer."
        assert k > 0, "k should be greater than 0."
        assert isinstance(metric, str), "Metric should be a string."
        valid_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
        assert metric in valid_metrics, f"Metric should be one of {valid_metrics}."
        if direction:
            assert direction in ['greater', 'less'], "Direction should be either 'greater' or 'less'."
        if prob_direction:
            assert prob_direction in ['increase', 'decrease'], "prob_direction should be either 'increase' or 'decrease'."
        if classification_type:
            assert classification_type in ['binary', 'probabilistic'], "classification_type should be either 'binary' or 'probabilistic'."
        
        self.instance = instance
        self.data = data
        self.labels = labels
        self.k = k
        self.metric = metric
        self.direction = direction
        self.prob_direction = prob_direction
        self.classification_type = classification_type
        self.scaler = MinMaxScaler()

    def fit(self):
        """Fit kNN model for prediction."""
        self.data_scaled = self.scaler.fit_transform(self.data)
        self.instance_scaled = self.scaler.transform([self.instance])
        
        self.is_classification = len(np.unique(self.labels)) <= 2
        
        if self.is_classification:
            logger.info("Fitting kNN classifier.")
            self.model = KNeighborsClassifier(n_neighbors=self.k, metric=self.metric)
        else:
            logger.info("Fitting kNN regressor.")
            self.model = KNeighborsRegressor(n_neighbors=self.k, metric=self.metric)
        
        self.model.fit(self.data_scaled, self.labels)
        self.instance_prediction = self.model.predict(self.instance_scaled)[0]
        
        self.original_proba_class_1 = None
        if self.is_classification and self.classification_type == 'probabilistic':
            logger.info("Calculating probability of class 1.")
            self.original_proba_class_1 = self.model.predict_proba(self.instance_scaled)[0][1]

        # if instance scaled in data_scaled return instance prediction as label
        if self.instance_scaled.astype(float) in self.data_scaled.astype(float):
            logger.warning("Instance is in the data. Returning instance prediction as label.")
            self.instance_prediction = self.labels[np.where(self.data_scaled == self.instance_scaled)[0][0]]
            self.original_proba_class_1 = self.labels[np.where(self.data_scaled == self.instance_scaled)[0][0]]

    def retrieve_counterfactuals(self):
        """Retrieve counterfactuals for the instance using the fitted model."""
        distances, indices = self.model.kneighbors(self.instance_scaled)
        counterfactuals = {}
        if self.is_classification:
            if self.classification_type == 'binary':
                for idx, dist in zip(indices[0], distances[0]):
                        if self.labels[idx] != self.instance_prediction:
                            counterfactuals[tuple(self.data[idx])] = {
                                "distance": dist,
                                "label": self.labels[idx]
                            }
                self.results =  {
                    "original_instance": self.instance,
                    "prediction": self.instance_prediction,
                    "counterfactuals": counterfactuals
                }
            elif self.classification_type == 'probabilistic':
                for idx, dist in zip(indices[0], distances[0]):
                        prob_class_1 = self.model.predict_proba([self.data_scaled[idx]])[0][1]
                        if (self.prob_direction == 'increase' and prob_class_1 > self.original_proba_class_1) or \
                            (self.prob_direction == 'decrease' and prob_class_1 < self.original_proba_class_1):
                            counterfactuals[tuple(self.data[idx])] = {
                                "distance": dist,
                                "probability_class_1": prob_class_1
                            }
                self.results  = {
                    "original_instance": self.instance,
                    "prediction": self.original_proba_class_1,
                    "counterfactuals": counterfactuals
                }
        else:
            for idx, dist in zip(indices[0], distances[0]):
                prediction = self.model.predict([self.data_scaled[idx]])[0]
                if (self.direction == 'greater' and prediction > self.instance_prediction) or \
                    (self.direction == 'less' and prediction < self.instance_prediction):
                    counterfactuals[tuple(self.data[idx])] = {
                        "distance": dist,
                        "prediction": prediction
                    }
            self.results = {
                "original_instance": self.instance,
                "prediction": self.instance_prediction,
                "counterfactuals": counterfactuals
            }

        return self.results
    
    def show_results(self):
        """Display the results."""
        logger.info('__'*30)
        logger.info(" ")
        logger.info("Results:")
        logger.info(f"----> Original instance: {self.results['original_instance']}")
        logger.info(f"----> Prediction: {self.results['prediction']}")
        logger.info(" ")
        logger.info('__'*30)
        if self.results["counterfactuals"] == {}:
            logger.info("No counterfactuals found.")
        else:
            for cf, details in self.results["counterfactuals"].items():
                logger.info(f"Counterfactual: {cf}")
                logger.info(f"Details:")
                for key, value in details.items():
                    logger.info(f"----> {key}: {value}")
                logger.info('__'*30)

if __name__ == "__main__":

    data = np.array([[3, 2.5], [2, 3], [3, 3], [6, 7], [7, 8], [8, 9], [10, 10], [11, 12], [12, 13]])
    labels = np.array([0, 1, 0, 1, 0, 0, 1, 1, 1])
    instance = np.array([3, 3])

    cf_generator = KNNCounterfactualGenerator(instance, data, labels, 
                                            k=8,
                                            metric='manhattan',
                                            classification_type='probabilistic',
                                            prob_direction='increase'
                                            )
    cf_generator.fit()
    cf_generator.retrieve_counterfactuals()
    cf_generator.show_results()
