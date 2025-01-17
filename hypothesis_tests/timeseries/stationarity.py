import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning
warnings.simplefilter('ignore', InterpolationWarning)

class CheckStationarity:
    """Check stationarity of a time series."""

    def __init__(self, data: np.ndarray):
        assert not np.isnan(data).any(), "Data should not contain any NaN values"
        assert len(data.shape) == 1, "Data should be a 1D array"
        self.data = data

    # return dictionary
    def adf_test(self) -> dict:
        """Perform Augmented Dickey-Fuller Test."""
        return adfuller(self.data)

    def kpss_test(self) -> dict:
        """Perform Kwiatkowski-Phillips-Schmidt-Shin Test."""
        return kpss(self.data)

    def evaluate_stationarity(self) -> dict:
        """Evaluate stationarity using ADF and KPSS tests."""
        # Perform ADF test
        adf_result = self.adf_test()
        results_adf = {
            "statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "critical_values": {key: float(value) for key, value in adf_result[4].items()}
        }

        # Perform KPSS test
        kpss_result = self.kpss_test()
        results_kpss = {
            "statistic": float(kpss_result[0]),
            "p_value": float(kpss_result[1]),
            "critical_values": {key: float(value) for key, value in kpss_result[3].items()}
        }

        return {
            "adf_test": results_adf,
            "kpss_test": results_kpss
        }

    def plot_stationarity_results(self) -> None:
        """Plot stationarity test results using heatmaps."""
        results = self.evaluate_stationarity()

        # Extract ADF and KPSS test results
        adf_results = results["adf_test"]
        kpss_results = results["kpss_test"]

        # Prepare dataframes for critical values
        df_adf = pd.DataFrame(adf_results["critical_values"].items(), columns=["Significance Level", "Critical Value"])
        df_kpss = pd.DataFrame(kpss_results["critical_values"].items(), columns=["Significance Level", "Critical Value"])
        df_kpss = df_kpss.sort_values(by="Significance Level", ascending=True)

        # Plot the results
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # ADF heatmap
        adf_cmap = ["red" if adf_results["statistic"] > value else "green" for value in df_adf["Critical Value"]]
        sns.heatmap(
            df_adf.set_index("Significance Level").T, 
            annot=True, fmt=".3f", cmap=adf_cmap, cbar=False, linewidths=0.5, ax=ax[0],
            alpha=0.5
        )
        ax[0].set_title(f'ADF Test: Statistic = {adf_results["statistic"]:.4f} (P-Value = {adf_results["p_value"]:.4f})')

        # KPSS heatmap
        kpss_cmap = ["red" if kpss_results["statistic"] > value else "green" for value in df_kpss["Critical Value"]]
        sns.heatmap(
            df_kpss.set_index("Significance Level").T, 
            annot=True, fmt=".3f", cmap=kpss_cmap, cbar=False, linewidths=0.5, ax=ax[1],
            alpha=0.5
        )
        ax[1].set_title(f'KPSS Test: Statistic = {kpss_results["statistic"]:.4f} (P-Value = {kpss_results["p_value"]:.4f})')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    
if __name__ == "__main__":

    # make time-series data with 1000 obs
    data = np.random.normal(0, 1, 1000)
    cs = CheckStationarity(data)
    print(cs.evaluate_stationarity())
    cs.plot_stationarity_results()


