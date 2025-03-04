import numpy as np
from scipy.stats import rankdata
from loguru import logger

class RankingMetrics:
    """Ranking Metrics for evaluating ranking quality."""

    def __init__(self, scores: np.ndarray, k: int, method: int) -> None:
        if not isinstance(scores, np.ndarray):
            raise TypeError("scores must be a numpy array")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if method not in {0, 1}:
            raise ValueError("method must be 0 or 1")
        self.scores = scores
        self.k = k
        self.ranks = self.compute_ranking()
        self.method = method
        
    def compute_ranking(self) -> np.ndarray:
        """Computes ranking of scores (the bigger the number the lower the rank)."""
        return rankdata(-self.scores)

    def _discounting(self, size: int) -> np.ndarray:
        """Computes discount values based on the method."""
        return np.log2(np.arange(2, size + 2 if method == 1 else size + 1))

    def _dcg_at_k(self, scores: np.ndarray) -> float:
        """Computes Discounted Cumulative Gain at K."""
        k_scores = np.asfarray(scores[:self.k])
        discounts = self._discounting(k_scores.size)
        if self.method == 0:
            return k_scores[0] + np.sum(k_scores[1:]/discounts[1:])
        return np.sum(k_scores/discounts)

    def ndcg_at_k(self) -> float:
        """Computes Normalized Discounted Cumulative Gain at K."""
        sorted_scores = np.sort(self.scores)[::-1]  # Sort in descending order
        ideal_dcg = self._dcg_at_k(sorted_scores)
        if ideal_dcg == 0:
            return 0.0
        dcg = self._dcg_at_k(self.scores)
        logger.debug(f'IDCG@K: {ideal_dcg}')
        logger.debug(f'DCG@K: {dcg}')
        logger.debug(f'NDCG@K: {dcg/ideal_dcg}')
        ndcg = dcg / ideal_dcg
        return ndcg
    
    def mean_reciprocal_rank(self) -> float:
        """Computes Mean Reciprocal Rank."""
        reciprocal_ranks = 1 / self.ranks
        mrr = np.mean(reciprocal_ranks)
        logger.debug(f'MRR: {mrr}')
        return mrr
        
    def evaluate(self) -> dict:
        """Evaluates ranking performance metrics."""
        return {'ndcg@k': self.ndcg_at_k(), 
                'mrr': self.mean_reciprocal_rank()}

if __name__ == "__main__":

    scores = np.array([1, 2, 3, 0, 0, 3, 2, 2, 3, 3])
    k = len(scores) - 2
    method = 1
    ranking_metrics = RankingMetrics(scores, k, method)
    print('bad score', ranking_metrics.evaluate())

    scores_better = np.array([3, 3, 2, 2, 1, 0, 0, 0, 0, 0])
    ranking_metrics_better = RankingMetrics(scores_better, k, method)
    print('good score', ranking_metrics_better.evaluate())