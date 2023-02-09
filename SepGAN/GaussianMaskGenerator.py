
import numpy as np
import os
import matplotlib.pyplot as plt

def gaussian1d(pattern_shape, factor,direction = "column",center=None, cov=None):
    """
    Description: creates a 1D gaussian sampling pattern either in the row or column direction
    of a 2D image
    :param factor: sampling factor in the desired direction
    :param direction: sampling direction, 'row' or 'column'
    :param pattern_shape: shape of the desired sampling pattern.
    :param center: coordinates of the center of the Gaussian distribution
    :param cov: covariance matrix of the distribution
    :return: sampling pattern image. It is a boolean image
    """
    if direction != "column":
        pattern_shape = (pattern_shape[1],pattern_shape[0])

    if center is None:
        center = np.array([1.0 * pattern_shape[1] / 2 - 0.5])

    if cov is None:
        cov = np.array([[(1.0 * pattern_shape[1] / 4) ** 2]])


    factor = int(factor * pattern_shape[1])

    samples = np.array([0])

    m = 1  # Multiplier. We have to increase this value
    # until the number of points (disregarding repeated points)
    # is equal to factor 

    while (samples.shape[0] < factor):

        samples = np.random.multivariate_normal(center, cov, m * factor)

        samples = np.rint(samples).astype(int)
        indexes = np.logical_and(samples >= 0, samples < pattern_shape[1])
        samples = samples[indexes]
        samples = np.unique(samples)
        if samples.shape[0] < factor:
            m *= 2
            continue

    indexes = np.arange(samples.shape[0], dtype=int)
    np.random.shuffle(indexes)
    samples = samples[indexes][:factor]
    under_pattern = np.zeros(pattern_shape, dtype=bool)
    under_pattern[:, samples] = True

    if direction != "column":
        under_pattern = under_pattern.T

    return under_pattern

# mask = gaussian1d([256,256],10)
# plt.figure()
# plt.imshow(mask, plt.cm.gray)
# plt.show()
if __name__ == '__main__':
    mask = gaussian1d([256,256],10)
    plt.figure()
    plt.imshow(mask, plt.cm.gray)
    plt.show()
    print("hello")