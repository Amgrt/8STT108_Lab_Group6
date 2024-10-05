import numpy as np
import matplotlib.pyplot as plt

# 1
# Generate Dataset
np.random.seed(0)
data = np.random.normal(0, 1, 100)


# Define Gaussian kernel function
def gaussian_kernel(x):
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


# Compute KDE
def kde(data, kernel, bandwidth, x_grid):
    n = len(data)
    kde_values = np.zeros_like(x_grid)
    for i, x in enumerate(x_grid):
        kde_values[i] = np.sum(kernel((x - data) / bandwidth))
    return kde_values / (n * bandwidth)


# Draw KDE and histogram
x_grid = np.linspace(-4, 4, 1000)
bandwidth = 0.5
kde_values = kde(data, gaussian_kernel, bandwidth, x_grid)

plt.figure(figsize=(8, 6))
plt.plot(x_grid, kde_values, label="KDE (Gaussian Kernel)")
plt.hist(data, bins=30, density=True, alpha=0.5, label="Histogram")
plt.title('Kernel Density Estimation with Gaussian Kernel')
plt.legend()
plt.show()


# 2
# Define other kernel functions
def epanechnikov_kernel(x):
    return 0.75 * (1 - x ** 2) * (np.abs(x) <= 1)


def uniform_kernel(x):
    return 0.5 * (np.abs(x) <= 1)


def triangular_kernel(x):
    return (1 - np.abs(x)) * (np.abs(x) <= 1)


# Drawing KDE with different kernel functions
kernels = {
    "Gaussian": gaussian_kernel,
    "Epanechnikov": epanechnikov_kernel,
    "Uniform": uniform_kernel,
    "Triangular": triangular_kernel
}

plt.figure(figsize=(10, 8))
for kernel_name, kernel in kernels.items():
    kde_values = kde(data, kernel, bandwidth, x_grid)
    plt.plot(x_grid, kde_values, label=f"KDE ({kernel_name} Kernel)")

plt.hist(data, bins=30, density=True, alpha=0.5, label="Histogram")
plt.title('Comparison of Different Kernels in KDE')
plt.legend()
plt.show()

# 3
# Fixed Gaussian kernel, using different bandwidths
bandwidths = [0.1, 0.5, 1, 2]

plt.figure(figsize=(8, 6))
for bw in bandwidths:
    kde_values = kde(data, gaussian_kernel, bw, x_grid)
    plt.plot(x_grid, kde_values, label=f"Bandwidth = {bw}")

plt.hist(data, bins=30, density=True, alpha=0.5, label="Histogram")
plt.title('Effect of Bandwidth on KDE (Gaussian Kernel)')
plt.legend()
plt.show()

# 分析：带宽越小，KDE估计的分布越尖锐，但方差更大；
# 带宽越大，估计的分布越平滑，但可能失去细节。

# 4
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
sepal_width = iris.data[:, 1]

x_grid = np.linspace(min(sepal_width), max(sepal_width), 1000)
bandwidth = 0.2
kde_values = kde(sepal_width, gaussian_kernel, bandwidth, x_grid)

plt.figure(figsize=(8, 6))
plt.plot(x_grid, kde_values, label="KDE (Gaussian Kernel)")
plt.hist(sepal_width, bins=30, density=True, alpha=0.5, label="Histogram")
plt.title('KDE for Sepal Width in Iris Dataset')
plt.legend()
plt.show()

# 解释：该图展示了鸢尾花数据集中花萼宽度的概率密度估计，
# 显示了花萼宽度的分布模式。
