import numpy as np

class ClassificationMetrics:
    " A class to calculate classification metrics "

    def __init__(self, decimals, y_true, y_pred):
        assert isinstance(decimals, int), "decimals must be an integer"
        assert len(y_true) == len(y_pred), "Length of y_true and y_pred should be same"
        # assert binary (0, 1) classification
        assert set(y_true) == {0, 1}, "y_true should be binary"
        assert set(y_pred) == {0, 1}, "y_pred should be binary"
        self.decimals = decimals
        self.y_true = y_true
        self.y_pred = y_pred

    @property
    def true_positive(self):
        """True Positive (TP)"""
        return round(int(np.sum(np.logical_and(self.y_true == 1, self.y_pred == 1))), self.decimals)

    @property
    def true_negative(self):
        """True Negative (TN)"""
        return round(int(np.sum(np.logical_and(self.y_true == 0, self.y_pred == 0))), self.decimals)

    @property
    def false_positive(self):
        """False Positive (FP)"""
        return int(np.sum(np.logical_and(self.y_true == 0, self.y_pred == 1)))

    @property
    def false_negative(self):
        """False Negative (FN)"""
        return int(np.sum(np.logical_and(self.y_true == 1, self.y_pred == 0)))
    
    def accuracy(self):
        " Accuracy "
        accuracy = (self.true_positive() + self.true_negative()) / len(self.y_true)
        return round(accuracy, self.decimals)
    
    def recall(self):
        " Recall "
        recall =  self.true_positive() / (self.true_positive() + self.false_negative())
        return round(recall, self.decimals)
    
    def precision(self):
        " Precision "
        precision = self.true_positive() / (self.true_positive() + self.false_positive())
        return round(precision, self.decimals)
    
    def f1_score(self):
        " F1 Score "
        f1_score =  2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        return round(f1_score, self.decimals)

    def evaluate_classifier(self):
        return {
            "f1": self.f1_score(),
            "accuracy": self.accuracy(),
        }
    
    def plot_ROC(self):
        pass
    

if __name__ == "__main__":
    decimals = 3
    y_true = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 0])
    cm = ClassificationMetrics(decimals, y_true, y_pred)
    print(cm.evaluate_classifier())