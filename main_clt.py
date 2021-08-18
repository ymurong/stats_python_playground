import random
import matplotlib.pyplot as plt
from core.descriptive_stats import mean


def sample(num_of_samples, sample_sz):
    data = []
    for _ in range(num_of_samples):
        data.append(mean([random.uniform(0.0, 1.0) for _ in range(sample_sz)]))
    return data


if __name__ == "__main__":
    data = sample(10000, 100)
    plt.hist(data, bins='auto', rwidth=0.8)
    plt.axvline(x=mean(data), c='red')
    plt.show()
