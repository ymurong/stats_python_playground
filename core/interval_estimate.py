from math import sqrt
from typing import List, Tuple
from core.descriptive_stats import mean, std, variance
from scipy.stats import norm, t, chi2, f


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


def mean_diff_ci_t_est(data1: List[float], data2: List[float], alpha: float, equal=True) -> Tuple[float, float]:
    """
    两个总体方差未知且相等或不等， 求均值差的置信区间
    Find the confidence interval of the difference between the means of two samples whose overall variances are unknown and equal or unequal
    """
    n1 = len(data1)
    n2 = len(data2)
    mean_diff = mean(data1) - mean(data2)

    sample1_var = variance(data1)
    sample2_var = variance(data2)

    if equal:
        Sw = sqrt(((n1 - 1) * sample1_var + (n2 - 1) * sample2_var) / (n1 + n2 - 2))
        t_value = abs(t.ppf(alpha / 2, n1 + n2 - 2))
        return mean_diff - Sw * sqrt(1 / n1 + 1 / n2) * t_value, \
               mean_diff + Sw * sqrt(1 / n1 + 1 / n2) * t_value
    else:
        df_numerator = (sample1_var / n1 + sample2_var / n2) ** 2
        df_denominator = (sample1_var / n1) ** 2 / (n1 - 1) + (sample2_var / n2) ** 2 / (n2 - 1)
        df = df_numerator / df_denominator
        t_value = abs(t.ppf(alpha / 2, df))
        return mean_diff - sqrt(sample1_var / n1 + sample2_var / n2) * t_value, \
               mean_diff + sqrt(sample1_var / n1 + sample2_var / n2) * t_value


def mean_diff_ci_z_est(data1: List[float], data2: List[float], alpha: float, sigma1: float, sigma2: float) -> Tuple[
    float, float]:
    """
    两个总体方差已知， 求均值差的置信区间
    Find the confidence interval of the difference between the means of two samples whose overall variances are known
    """
    n1 = len(data1)
    n2 = len(data2)
    mean_diff = mean(data1) - mean(data2)
    z_value = abs(norm.ppf(alpha / 2))
    return mean_diff - sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2) * z_value, \
           mean_diff + sqrt(sigma1 ** 2 / n1 + sigma2 ** 2 / n2) * z_value


def var_ratio_ci_est(data1: List[float], data2: List[float], alpha: float) -> Tuple[float, float]:
    """
    两个总体方差未知， 求方差比的置信区间
    Find the confidence interval of the difference between the means of two samples whose overall variances are known
    """
    n1 = len(data1)
    n2 = len(data2)
    f_lower_value = f.ppf(alpha / 2, n1 - 1, n2 - 1)
    f_upper_value = f.ppf(1 - alpha / 2, n1 - 1, n2 - 1)
    var_ratio = variance(data1) / variance(data2)
    return var_ratio / f_upper_value, var_ratio / f_lower_value
