import numpy as np
from regression_score import RegressionMetrics
from loguru import logger

class ProbabilisticForecastMetrics(RegressionMetrics):
    " Probabilistic Forecast Metrics "

    alpha = .2
    
    def __init__(self, decimals:int, y_trues:np.array, y_preds:np.array, quantile_10_preds=None, quantile_90_preds=None) -> None:
        super().__init__(decimals, y_trues, y_preds)
        self.quantile_10_preds = quantile_10_preds
        self.quantile_90_preds = quantile_90_preds

    def pinball_loss(self, quantile: float) -> float:
        " Pinball Loss "
        assert quantile > 0 and quantile < 1, "Quantile must be between 0 and 1"
        pinball =  np.mean(np.maximum(quantile * (self.y_trues - self.y_preds), (quantile - 1) * (self.y_trues - self.y_preds)))
        return round(pinball, self.decimals)
    
    def coverage_interval(self) -> float:
        " Coverage Interval "
        cov_interval = np.mean((self.y_trues >= self.quantile_10_preds) & (self.y_trues <= self.quantile_90_preds))
        return round(cov_interval, self.decimals)
    
    def sharpness(self) -> float:
        " Sharpness "
        sharp =  np.mean(self.quantile_90_preds - self.quantile_10_preds)
        return round(sharp, self.decimals)

    def _winkler_score_single(self, y_true:float, q10:float, q90:float) -> float:
        """
        Compute the Winkler Score for a single instance.
        """
        interval_width = abs(q90 - q10)
        score = interval_width
        if y_true < min(q90, q10):
            score += (2 / self.alpha) * (min(q90, q10) - y_true)
        elif y_true > max(q90, q10):
            score += (2 / self.alpha) * (y_true - max(q90, q10))
        return score

    def winkler_score(self) -> float:
        " Winkler Score "
        v_winkler_score = np.vectorize(self._winkler_score_single)
        scores = v_winkler_score(self.y_trues, self.quantile_10_preds, self.quantile_90_preds)
        winkler= np.mean(scores)
        return round(winkler, self.decimals)
    
    def evaluate_forecaster(self) -> dict:
        " Evaluate all metrics "
        metrics = super().evaluate_regressor()
        metrics['coverage_interval'] = self.coverage_interval()
        metrics['sharpness'] = self.sharpness()
        metrics['pinball_loss_10'] = self.pinball_loss(0.1)
        metrics['pinball_loss_90'] = self.pinball_loss(0.9)
        metrics['winkler_score'] = self.winkler_score()
        for key, value in metrics.items():
            logger.info(f"{key}: {value}")
        return metrics


if __name__ == "__main__":

    decimals = 4

    y_preds = np.array([1, 2, 3, 4, 5])
    y_trues = np.array([2, 2, 3, 4, 5])

    quantile_10_preds = np.array([1.1, 2.1, 3.1, 4.1, 5.1])
    quantile_90_preds = np.array([1.9, 2.9, 3.9, 4.9, 5.9])

    metrics = ProbabilisticForecastMetrics(decimals, y_trues, y_preds, quantile_10_preds, quantile_90_preds)
