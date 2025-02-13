from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

class TTests:
    """ 
    Hypothesis tests using t-test
    """
    def __init__(self, data) -> None:
        self.data = data

    def ttest_1samp_equal(self, popmean: float) -> dict:
        """
        One sample t-test
        statistic = (np.mean(a) - popmean)/se
        """
        return ttest_1samp(self.data, popmean)
    
    def ttest_1samp_greater(self, popmean):
        """
        One sample t-test
        statistic = (np.mean(a) - popmean)/se
        """
        return ttest_1samp(self.data, popmean, alternative='greater')
    
    def ttest_1samp_less(self, popmean):
        """
        One sample t-test
        statistic = (np.mean(a) - popmean)/se
        """
        return ttest_1samp(self.data, popmean, alternative='less')

    def ttest_ind(self, data2):
        """
        Independent samples t-test
        """
        return ttest_ind(self.data, data2)
    
    def welchs_ttest(self, data2):
        """
        Welch's t-test
        """
        return ttest_ind(self.data, data2, equal_var=False)

    def ttest_rel(self, data2):
        """
        Paired samples t-test
        """
        return ttest_rel(self.data, data2)