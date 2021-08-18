from math import sqrt
from typing import List, Tuple
from core.descriptive_stats import mean, std, variance
from scipy.stats import norm, t, chi2


def mean_ci_est(data: List[float], alpha: float, sigma: float = None) -> Tuple[float, float]:
    """
    sigma is known/unknown, calculate confidence interval of miu
    """
    n = len(data)
    sample_mean = mean(data)

    if sigma is None:
        # we don't have the std of population
        s = std(data)
        se = s / sqrt(n)
        t_value = abs(t.ppf(alpha / 2, n - 1))
        return sample_mean - se * t_value, sample_mean + se * t_value
    else:
        # we have the std of population
        se = sigma / sqrt(n)
        z_value = abs(norm.ppf(alpha / 2))  # norm.ppf returns quantile corresponding to the lower tail probability q
        return sample_mean - se * z_value, sample_mean + se * z_value


def var_ci_est(data: List[float], alpha: float) -> Tuple[float, float]:
    """
    miu is unknown, calculate confidence interval of sigma**2
    """
    n = len(data)
    s2 = variance(data)
    chi2_lower_value = chi2.ppf(alpha / 2, n - 1)
    chi2_upper_value = chi2.ppf(1 - alpha / 2, n - 1)
    return (n - 1) * s2 / chi2_upper_value, (n - 1) * s2 / chi2_lower_value
