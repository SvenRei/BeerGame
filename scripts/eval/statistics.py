import numpy as np
from scipy import stats

def compute_robustness_metrics(costs):
    """
    Returns mean, std, and 95% CVaR of a cost distribution.
    """
    costs = np.sort(costs)
    mean = np.mean(costs)
    std = np.std(costs)
    # CVaR at 95% is the mean of the top 5% worst cases
    cvar_95 = np.mean(costs[int(0.95 * len(costs)):])
    return mean, std, cvar_95

def compare_policies(costs_ai, costs_baseline):
    """
    Returns p-value and Cohen's d for statistical proof.
    """
    t_stat, p_val = stats.ttest_ind(costs_ai, costs_baseline, equal_var=False)
    
    # Calculate Cohen's d
    n1, n2 = len(costs_ai), len(costs_baseline)
    var1, var2 = np.var(costs_ai, ddof=1), np.var(costs_baseline, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = (np.mean(costs_baseline) - np.mean(costs_ai)) / pooled_std
    
    return p_val, cohens_d