from core.descriptive_stats import mean, variance
from typing import List, Callable
import random
import matplotlib.pyplot as plt


def variance_bias(data: List[float]) -> float:
    n = len(data)
    if n <= 1:
        return None

    mean_value = mean(data)
    return sum([(e - mean_value) ** 2 for e in data]) / n


def sample(num_of_samples: int, sample_size: int, var: Callable[[List[float]], float]) -> List[float]:
    """
    :param num_of_samples:
    :param sample_size:
    :param var: variance calculation function
    :return: list of sample variance
    """
    data = []
    for _ in range(num_of_samples):
        data.append(
            var([random.uniform(0.0, 1.0) for _ in range(sample_size)]))  # random.uniform(0.0, 1.0) -> 从0到1的均匀分布中获得一个数
    return data


if __name__ == '__main__':
    data1 = sample(1000, 40, variance_bias)
    plt.hist(data1, bins='auto', alpha=0.5, rwidth=0.8, label='biased sample variance')
    plt.axvline(x=mean(data1), c='blue', label='biased sample variance')
    print("bias : ", mean(data1), 1/12)

    data2 = sample(1000, 40, variance)
    plt.hist(data2, bins='auto', alpha=0.5, rwidth=0.8, label='unbiased sample variance')
    plt.axvline(x=mean(data2), c='green', label='unbiased sample variance')
    print("unbias : ", mean(data2), 1/12)

    plt.axvline(x=(1 - 0) ** 2 / 12, c='red', label='population variance')
    plt.legend(loc='upper right')
    plt.title('unbiased/biased sample variance')
    plt.show()
