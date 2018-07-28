import loadSPEfiles as lf
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lfilter


def draw_normal_plot(x, y):
    plt.figure(1)
    plt.plot(x, y)
    plt.show(block=False)


def draw_scatter_graph(x, y):
    plt.figure(2)
    plt.scatter(x, y, s=1, alpha=0.10, marker='o')
    plt.show(block=False)


def draw_noise_cancelling(x, y, n):
    b = [1.0 / n] * n
    yprime = lfilter(b, 1, y)
    plt.figure(3)
    plt.plot(x, yprime, marker='v', markevery=np.ndarray.tolist(find_local_maxs(x, yprime)))
    plt.show()


def find_local_maxs(x, y):
    maxs = np.array([0])
    for r in range(2, x.size):
        if y[r] < y[r - 1] and y[r - 1] > y[r - 2]:
            maxs = np.append(maxs, r)
    print(maxs)
    return maxs


if __name__ == '__main__':
    data = lf.load('Data\WS2 reflection spectra[130]\WS2 reflection spectra\\20180404 WS2_1 d.spe')

    wavelengths = data[0]
    intensities = data[1]

    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)

    print('Wavelength\t\t\t\tIntensity')
    for i in range(0, wavelengths.size):
        print(str(wavelengths[i]) + '\t\t' + str(intensities[i]))
    draw_normal_plot(wavelengths, intensities)
    draw_scatter_graph(wavelengths, intensities)
    draw_noise_cancelling(wavelengths, intensities, 40)
