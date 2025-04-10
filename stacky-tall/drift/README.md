# Drift Detection

### Types of Dataset Shift

- **Covariate or Input Drift**:  
  $$P(Y \mid X) = P_{\text{ref}}(Y \mid X) \quad \text{but} \quad P(X) \ne P_{\text{ref}}(X)$$

- **Label Drift**:  
  $$P(X \mid Y) = P_{\text{ref}}(X \mid Y) \quad \text{but} \quad P(Y) \ne P_{\text{ref}}(Y)$$

- **Concept Drift**:  
  $$P(Y \mid X) \ne P_{\text{ref}}(Y \mid X)$$

## Covariate drift
* Maximum Mean Discrepancy Two-Sample Test([Paper](https://jmlr.csail.mit.edu/papers/v13/gretton12a.html)) with p-value

### Example
#### Data before and after drift
<img src="imgs/covariate_drift.png" style="vertical-align: middle;">

#### Statistcial significance `p_value = np.mean(mmd_perms >= mmd_obs)`
<img src="imgs/mmd_permutations.png" style="vertical-align: middle;">