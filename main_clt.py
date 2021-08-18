import random
import matplotlib.pyplot as plt
from typing import List
from core.descriptive_stats import mean


def sample(num_of_samples: int, sample_size: int) -> List[float]:
    data = []
    for _ in range(num_of_samples):
        data.append(
            mean([random.uniform(0.0, 1.0) for _ in range(sample_size)]))  # random.uniform(0.0, 1.0) -> 从0到1的均匀分布中获得一个数
    return data


if __name__ == '__main__':
    data = sample(10000, 100)
    plt.hist(data, bins='auto', rwidth=0.8)
    plt.axvline(x=mean(data), c='red')
    plt.show()
