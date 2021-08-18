from collections import Counter
from math import sqrt
from typing import List


def frequency(data: List[float]) -> float:
    """频率"""
    counter = Counter(data)
    ret = []
    for point in counter.most_common():
        ret.append((point[0], point[1] / len(data)))
    return ret


def mode(data: List[float]) -> float:
    """众数"""
    counter = Counter(data)
    if counter.most_common()[0][1] == 1:
        return None, None

    count = counter.most_common()[0][1]
    ret = []
    for point in counter.most_common():
        if point[1] == count:
            ret.append(point[0])
        else:
            break
    return ret, count


def median(data: List[float]) -> float:
    """中位数"""
    sorted_data = sorted(data)
    n = len(sorted_data)

    if n % 2 == 1:
        return sorted_data[n // 2]

    return (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2


def mean(data: List[float]) -> float:
    """均值"""
    return sum(data) / len(data)


def rng(data: List[float]) -> float:
    """极差"""
    return max(data) - min(data)


def quartile(data: List[float]) -> float:
    """四分位数"""
    n = len(data)
    q1, q2, q3 = None, None, None
    if n >= 4:
        sorted_data = sorted(data)
        q2 = median(sorted_data)
        if n % 2 == 1:
            q1 = median(sorted_data[:n // 2])
            q3 = median(sorted_data[n // 2 + 1:])
        else:
            q1 = median(sorted_data[:n // 2])
            q3 = median(sorted_data[n // 2:])

    return q1, q2, q3


def variance(data: List[float]) -> float:
    """方差"""
    n = len(data)
    if n <= 1:
        return None

    mean_value = mean(data)
    return sum((e - mean_value) ** 2 for e in data) / (n - 1)


def std(data: List[float]) -> float:
    """标准差"""
    return sqrt(variance(data))
