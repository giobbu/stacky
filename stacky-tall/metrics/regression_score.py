
import numpy as np
from loguru import logger

class RegressionMetrics:
    " Regression Metrics "

    def __init__(self, decimals: int, y_trues: np.ndarray, y_preds: np.ndarray) -> None:
        assert isinstance(decimals, int), "decimals must be an integer"
        assert isinstance(y_trues, np.ndarray), "y_trues must be a numpy array"
        assert isinstance(y_preds, np.ndarray), "y_preds must be a numpy array"
        assert len(y_trues) == len(y_preds), "y_trues and y_preds must have the same length"
        self.decimals = decimals
        self.y_trues = y_trues
        self.y_preds = y_preds

    def rmse(self) -> float:
        " Root Mean Squared Error "
        rmse =  np.sqrt(np.mean((self.y_trues - self.y_preds) ** 2)) 
        return round(rmse, self.decimals)
    
    def mae(self) -> float:
        " Mean Absolute Error "
        mae = np.mean(np.abs(self.y_trues - self.y_preds))
        return round(mae, self.decimals)

    def mape(self) -> float:
        " Mean Absolute Percentage Error "
        assert np.all(self.y_trues != 0), "y_trues must not contain zeros"
        mape = np.mean(np.abs((self.y_trues - self.y_preds) / self.y_trues)) * 100
        return int(mape)
    
    def nrmse(self) -> float:
        " Normalized Root Mean Squared Error "
        nrmse = self.rmse() / (np.max(self.y_trues) - np.min(self.y_trues))
        return round(nrmse, self.decimals)
    
    def r2(self) -> float:
        " R^2 Score "
        r2 = 1 - np.sum((self.y_trues - self.y_preds) ** 2) / np.sum((self.y_trues - np.mean(self.y_trues)) ** 2)
        return round(r2, self.decimals)
    
    def evaluate_regressor(self) -> dict:
        " Evaluate all metrics "
        metrics = {
                'rmse': self.rmse(),
                'mae': self.mae(),
                'mape': self.mape(),
                'r2': self.r2(),
                'nrmse': self.nrmse()
            }
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        return metrics


if __name__ == "__main__":

    decimals = 4

    y_preds = np.array([1, 2, 3, 4, 5])
    y_trues = np.array([2, 2, 3, 4, 5])

    metrics = RegressionMetrics(decimals, y_trues, y_preds)


